type f16 = f16

-- Affin coupling forward: soronkent x1=row[0..half], x2=row[half..d]
-- scale = exp(clip(W_s * x2 + b_s)), y1 = x1*scale, t = W_t*y1 + b_t, y2 = x2+t
entry rsf_forward [n][d] (input: [n][d]f16)
  (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16)
  (s_bias: [d/2]f16) (t_bias: [d/2]f16)
  (clip_min: f16) (clip_max: f16) : *[n][d]f16 =
  let half = d / 2
  in map (\row ->
    let x1 = row[0:half]
    let x2 = row[half:d]
    -- Scale branch
    let scale = map (\j ->
      let dot = f16.sum (map2 (f16.*) weights_s[j] x2)
      let pre = s_bias[j] f16.+ dot
      let clipped = f16.max clip_min (f16.min clip_max pre)
      in f16.exp clipped
    ) (iota half)
    -- y1 = x1 * scale
    let y1 = map2 (f16.*) x1 scale
    -- Translation branch
    let trans = map (\j ->
      let dot = f16.sum (map2 (f16.*) weights_t[j] y1)
      in t_bias[j] f16.+ dot
    ) (iota half)
    -- y2 = x2 + trans
    let y2 = map2 (f16.+) x2 trans
    in y1 ++ y2
  ) input

-- Affin coupling backward: kimenetbol rekonstrualja a bemenetet, majd kiszamolja a gradienseket
-- Visszafejtés: x2 = y2 - t(y1), pre_s = W_s*x2 + b_s, scale = exp(clip(pre_s)), x1 = y1/scale
entry rsf_grad [n][d] (y_out: [n][d]f16) (dy: [n][d]f16)
  (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16)
  (s_bias: [d/2]f16) (t_bias: [d/2]f16)
  (clip_min: f16) (clip_max: f16)
  : (*[d/2][d/2]f16, *[d/2][d/2]f16, *[d/2]f16, *[d/2]f16) =
  let half = d / 2
  -- Visszafejtes es gradiensek kiszamitasa soronkent
  let per_row = map2 (\row dy_row ->
    let y1 = row[0:half]
    let y2 = row[half:d]
    let dy1 = dy_row[0:half]
    let dy2 = dy_row[half:d]
    -- x2 visszafejtes: x2 = y2 - W_t*y1 - b_t
    let x2 = map (\j ->
      let dot = f16.sum (map2 (f16.*) weights_t[j] y1)
      in y2[j] f16.- t_bias[j] f16.- dot
    ) (iota half)
    -- pre_s visszafejtes es scale
    let pre_s = map (\j ->
      let dot = f16.sum (map2 (f16.*) weights_s[j] x2)
      in s_bias[j] f16.+ dot
    ) (iota half)
    let scale = map (\j ->
      let clipped = f16.max clip_min (f16.min clip_max pre_s[j])
      in f16.exp clipped
    ) (iota half)
    -- x1 visszafejtes
    let x1 = map2 (\yv sv -> yv f16./ sv) y1 scale
    -- dy2 atadodik dx2-be
    let dx2_contrib = dy2
    -- ds: gradient scale-re (clip-elt tartomanyban: ds_j = dy1_j * y1_j, kulonben 0)
    let ds = map (\j ->
      if pre_s[j] f16.< clip_min f16.|| pre_s[j] f16.> clip_max
      then f16.i32 0
      else dy1[j] f16.* y1[j]
    ) (iota half)
    -- grad W_s: ds_j * x2_k
    let gws = map (\j ->
      map (\k -> ds[j] f16.* x2[k]) (iota half)
    ) (iota half)
    -- grad b_s
    let gbs = ds
    -- dx2 vegso = dy2 + sum_j(gws_jk * ds_j) -- W_s^T * ds
    let dx2 = map (\k ->
      dx2_contrib[k] f16.+ f16.sum (map (\j -> weights_s[j][k] f16.* ds[j]) (iota half))
    ) (iota half)
    -- dy1_total = dy1 * scale
    let dy1_total = map2 (f16.*) dy1 scale
    -- grad W_t: dy1_total_j * y1_k
    let gwt = map (\j ->
      map (\k -> dy1_total[j] f16.* x1[k]) (iota half)
    ) (iota half)
    -- grad b_t
    let gbt = dy1_total
    let _ = dx2
    in (gws, gwt, gbs, gbt)
  ) y_out dy
  -- Osszeadjuk a gradienseket
  let zero_mat = replicate (d/2) (replicate (d/2) (f16.i32 0))
  let zero_vec = replicate (d/2) (f16.i32 0)
  let (grad_s, grad_t, grad_bs, grad_bt) =
    reduce (\(a1,b1,c1,e1) (a2,b2,c2,e2) ->
      (map2 (map2 (f16.+)) a1 a2,
       map2 (map2 (f16.+)) b1 b2,
       map2 (f16.+) c1 c2,
       map2 (f16.+) e1 e2)
    ) (zero_mat, zero_mat, zero_vec, zero_vec) per_row
  in (grad_s, grad_t, grad_bs, grad_bt)

entry sfd_update [d] (weights: *[d][d]f16) (gradients: [d][d]f16) (learning_rate: f16) (momentum: f16) (velocity: *[d][d]f16) : (*[d][d]f16, *[d][d]f16) =
  let new_velocity = map2 (map2 (\v g -> momentum f16.* v f16.+ learning_rate f16.* g)) velocity gradients
  let new_weights = map2 (map2 (\w v -> w f16.- v)) weights new_velocity
  in (new_weights, new_velocity)

entry compute_loss [n][d] (output: [n][d]f16) (target: [n][d]f16) : f16 =
  let squared_diff = map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t))) output target
  let total = f16.sum (flatten squared_diff)
  let count = f16.i64 (n * d)
  in total f16./ count

entry accumulate_gradients [d] (grad1: *[d][d]f16) (grad2: [d][d]f16) : *[d][d]f16 =
  map2 (map2 (f16.+)) grad1 grad2

entry batch_forward [batch_size][seq_len][d] (inputs: [batch_size][seq_len][d]f16)
  (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16)
  (s_bias: [d/2]f16) (t_bias: [d/2]f16)
  (clip_min: f16) (clip_max: f16) : *[batch_size][seq_len][d]f16 =
  map (\sample -> rsf_forward sample weights_s weights_t s_bias t_bias clip_min clip_max) inputs

entry batch_gradients [batch_size][seq_len][d]
  (y_outs: [batch_size][seq_len][d]f16)
  (grad_outputs: [batch_size][seq_len][d]f16)
  (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16)
  (s_bias: [d/2]f16) (t_bias: [d/2]f16)
  (clip_min: f16) (clip_max: f16)
  : (*[d/2][d/2]f16, *[d/2][d/2]f16, *[d/2]f16, *[d/2]f16) =
  let per_sample = map2 (\sample dy_sample ->
    rsf_grad sample dy_sample weights_s weights_t s_bias t_bias clip_min clip_max
  ) y_outs grad_outputs
  let half = d / 2
  let zero_mat = replicate half (replicate half (f16.i32 0))
  let zero_vec = replicate half (f16.i32 0)
  in reduce (\(a1,b1,c1,e1) (a2,b2,c2,e2) ->
    (map2 (map2 (f16.+)) a1 a2,
     map2 (map2 (f16.+)) b1 b2,
     map2 (f16.+) c1 c2,
     map2 (f16.+) e1 e2)
  ) (zero_mat, zero_mat, zero_vec, zero_vec) per_sample

entry xavier_fill_inplace [d] (weights: *[d][d]f16) (seed: i32) : *[d][d]f16 =
  let scale = f16.sqrt (f16.f32 2.0 f16./ f16.i64 d)
  in map (\i ->
    map (\j ->
      let hash = (seed + i32.i64 i * 73856093 + i32.i64 j * 19349663) % 1000000
      let normalized = (f16.i32 hash) f16./ (f16.i32 1000000) f16.- f16.f32 0.5
      in normalized f16.* scale
    ) (iota d)
  ) (iota d)

entry batch_compute_loss [batch_size][seq_len][d] (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16) : f16 =
  let squared_diff = map2 (map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t)))) outputs targets
  let total = f16.sum (flatten (flatten squared_diff))
  let count = f16.i64 (batch_size * seq_len * d)
  in total f16./ count

entry scale_weights_inplace [d] (weights: *[d][d]f16) (scale_factor: f16) : *[d][d]f16 =
  map (map (\w -> w f16./ scale_factor)) weights

entry training_step [batch_size][seq_len][d]
  (inputs: [batch_size][seq_len][d]f16)
  (targets: [batch_size][seq_len][d]f16)
  (weights_s: *[d/2][d/2]f16)
  (weights_t: *[d/2][d/2]f16)
  (s_bias: *[d/2]f16)
  (t_bias: *[d/2]f16)
  (velocity_s: *[d/2][d/2]f16)
  (velocity_t: *[d/2][d/2]f16)
  (learning_rate: f16)
  (momentum: f16)
  (clip_min: f16)
  (clip_max: f16) : (*[d/2][d/2]f16, *[d/2][d/2]f16, *[d/2]f16, *[d/2]f16, *[d/2][d/2]f16, *[d/2][d/2]f16, f16) =
  let half = d / 2
  let outputs = batch_forward inputs weights_s weights_t s_bias t_bias clip_min clip_max
  let loss = batch_compute_loss outputs targets
  let grad_outputs = map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets
  let (grad_s, grad_t, grad_bs, grad_bt) = batch_gradients outputs grad_outputs weights_s weights_t s_bias t_bias clip_min clip_max
  let (new_weights_s, new_velocity_s) = sfd_update (weights_s :> *[half][half]f16) grad_s learning_rate momentum (velocity_s :> *[half][half]f16)
  let (new_weights_t, new_velocity_t) = sfd_update (weights_t :> *[half][half]f16) grad_t learning_rate momentum (velocity_t :> *[half][half]f16)
  -- Bias frissitese egyszeruen SGD-vel (momentum nelkul, egyszerusitett)
  let new_s_bias = map2 (\b g -> b f16.- learning_rate f16.* g) s_bias grad_bs
  let new_t_bias = map2 (\b g -> b f16.- learning_rate f16.* g) t_bias grad_bt
  in (new_weights_s, new_weights_t, new_s_bias, new_t_bias, new_velocity_s, new_velocity_t, loss)
