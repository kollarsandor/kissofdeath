type f16 = f16

let affin_forward_row [d] (row: [d]f16) (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16) (s_bias: [d/2]f16) (t_bias: [d/2]f16) (clip_min: f16) (clip_max: f16) : [d]f16 =
  let half = d / 2
  let x1 = row[0:half] :> [half]f16
  let x2 = row[half:d] :> [half]f16
  let scale = map (\j ->
    let dot = f16.sum (map2 (f16.*) weights_s[j] x2)
    let pre = s_bias[j] f16.+ dot
    let clipped = f16.max clip_min (f16.min clip_max pre)
    in f16.exp clipped
  ) (iota half)
  let y1 = map2 (f16.*) x1 scale
  let trans = map (\j ->
    let dot = f16.sum (map2 (f16.*) weights_t[j] y1)
    in t_bias[j] f16.+ dot
  ) (iota half)
  let y2 = map2 (f16.+) x2 trans
  in (y1 ++ y2) :> [d]f16

entry rsf_forward [n][d] (input: [n][d]f16) (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16) (s_bias: [d/2]f16) (t_bias: [d/2]f16) (clip_min: f16) (clip_max: f16) : *[n][d]f16 =
  map (\row -> affin_forward_row row weights_s weights_t s_bias t_bias clip_min clip_max) input

let affin_grad_row [d] (y_row: [d]f16) (dy_row: [d]f16) (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16) (s_bias: [d/2]f16) (t_bias: [d/2]f16) (clip_min: f16) (clip_max: f16) : ([d/2][d/2]f16, [d/2][d/2]f16, [d/2]f16, [d/2]f16) =
  let half = d / 2
  let y1 = y_row[0:half] :> [half]f16
  let y2 = y_row[half:d] :> [half]f16
  let dy1 = dy_row[0:half] :> [half]f16
  let dy2 = dy_row[half:d] :> [half]f16
  let x2 = map (\j ->
    let dot = f16.sum (map2 (f16.*) weights_t[j] y1)
    in y2[j] f16.- t_bias[j] f16.- dot
  ) (iota half)
  let pre_s = map (\j ->
    let dot = f16.sum (map2 (f16.*) weights_s[j] x2)
    in s_bias[j] f16.+ dot
  ) (iota half)
  let scale = map (\j ->
    let clipped = f16.max clip_min (f16.min clip_max pre_s[j])
    in f16.exp clipped
  ) (iota half)
  let x1 = map2 (\yv sv -> yv f16./ sv) y1 scale
  let ds = map (\j ->
    if pre_s[j] f16.< clip_min f16.|| pre_s[j] f16.> clip_max
    then f16.i32 0
    else dy1[j] f16.* y1[j]
  ) (iota half)
  let gws = map (\j -> map (\k -> ds[j] f16.* x2[k]) (iota half)) (iota half)
  let gbs = copy ds
  let dy1_total = map2 (f16.*) dy1 scale
  let gwt = map (\j -> map (\k -> dy1_total[j] f16.* x1[k]) (iota half)) (iota half)
  let gbt = copy dy1_total
  let _ = dy2
  in (gws, gwt, gbs, gbt)

entry rsf_grad [n][d] (y_out: [n][d]f16) (dy: [n][d]f16) (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16) (s_bias: [d/2]f16) (t_bias: [d/2]f16) (clip_min: f16) (clip_max: f16) : (*[d/2][d/2]f16, *[d/2][d/2]f16, *[d/2]f16, *[d/2]f16) =
  let half = d / 2
  let per_row = map2 (\row dy_row -> affin_grad_row row dy_row weights_s weights_t s_bias t_bias clip_min clip_max) y_out dy
  let zero_mat = replicate half (replicate half (f16.i32 0)) :> [half][half]f16
  let zero_vec = replicate half (f16.i32 0) :> [half]f16
  let (grad_s, grad_t, grad_bs, grad_bt) =
    reduce (\(a1, b1, c1, e1) (a2, b2, c2, e2) ->
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

entry batch_forward [batch_size][seq_len][d] (inputs: [batch_size][seq_len][d]f16) (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16) (s_bias: [d/2]f16) (t_bias: [d/2]f16) (clip_min: f16) (clip_max: f16) : *[batch_size][seq_len][d]f16 =
  map (\sample -> map (\row -> affin_forward_row row weights_s weights_t s_bias t_bias clip_min clip_max) sample) inputs

entry batch_gradients [batch_size][seq_len][d] (y_outs: [batch_size][seq_len][d]f16) (grad_outputs: [batch_size][seq_len][d]f16) (weights_s: [d/2][d/2]f16) (weights_t: [d/2][d/2]f16) (s_bias: [d/2]f16) (t_bias: [d/2]f16) (clip_min: f16) (clip_max: f16) : (*[d/2][d/2]f16, *[d/2][d/2]f16, *[d/2]f16, *[d/2]f16) =
  let half = d / 2
  let flat_y = flatten y_outs
  let flat_dy = flatten grad_outputs
  let per_row = map2 (\row dy_row -> affin_grad_row row dy_row weights_s weights_t s_bias t_bias clip_min clip_max) flat_y flat_dy
  let zero_mat = replicate half (replicate half (f16.i32 0)) :> [half][half]f16
  let zero_vec = replicate half (f16.i32 0) :> [half]f16
  in reduce (\(a1, b1, c1, e1) (a2, b2, c2, e2) ->
    (map2 (map2 (f16.+)) a1 a2,
     map2 (map2 (f16.+)) b1 b2,
     map2 (f16.+) c1 c2,
     map2 (f16.+) e1 e2)
  ) (zero_mat, zero_mat, zero_vec, zero_vec) per_row

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

entry training_step [batch_size][seq_len][d] (inputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16) (weights_s: *[d/2][d/2]f16) (weights_t: *[d/2][d/2]f16) (s_bias: *[d/2]f16) (t_bias: *[d/2]f16) (velocity_s: *[d/2][d/2]f16) (velocity_t: *[d/2][d/2]f16) (learning_rate: f16) (momentum: f16) (clip_min: f16) (clip_max: f16) : (*[d/2][d/2]f16, *[d/2][d/2]f16, *[d/2]f16, *[d/2]f16, *[d/2][d/2]f16, *[d/2][d/2]f16, f16) =
  let half = d / 2
  let outputs = map (\sample -> map (\row -> affin_forward_row row weights_s weights_t s_bias t_bias clip_min clip_max) sample) inputs
  let loss = batch_compute_loss outputs targets
  let grad_outputs = map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets
  let flat_y = flatten outputs
  let flat_dy = flatten grad_outputs
  let zero_mat = replicate half (replicate half (f16.i32 0)) :> [half][half]f16
  let zero_vec = replicate half (f16.i32 0) :> [half]f16
  let per_row = map2 (\row dy_row -> affin_grad_row row dy_row weights_s weights_t s_bias t_bias clip_min clip_max) flat_y flat_dy
  let (grad_s, grad_t, grad_bs, grad_bt) =
    reduce (\(a1, b1, c1, e1) (a2, b2, c2, e2) ->
      (map2 (map2 (f16.+)) a1 a2,
       map2 (map2 (f16.+)) b1 b2,
       map2 (f16.+) c1 c2,
       map2 (f16.+) e1 e2)
    ) (zero_mat, zero_mat, zero_vec, zero_vec) per_row
  let new_velocity_s = map2 (map2 (\v g -> momentum f16.* v f16.+ learning_rate f16.* g)) velocity_s grad_s
  let new_weights_s = map2 (map2 (\w v -> w f16.- v)) weights_s new_velocity_s
  let new_velocity_t = map2 (map2 (\v g -> momentum f16.* v f16.+ learning_rate f16.* g)) velocity_t grad_t
  let new_weights_t = map2 (map2 (\w v -> w f16.- v)) weights_t new_velocity_t
  let new_s_bias = map2 (\b g -> b f16.- learning_rate f16.* g) s_bias grad_bs
  let new_t_bias = map2 (\b g -> b f16.- learning_rate f16.* g) t_bias grad_bt
  in (new_weights_s, new_weights_t, new_s_bias, new_t_bias, new_velocity_s, new_velocity_t, loss)
