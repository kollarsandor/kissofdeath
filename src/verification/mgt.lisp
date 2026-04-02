(in-package "ACL2")

(defun all-natp (x)
  (declare (xargs :guard t))
  (if (atom x)
      (equal x nil)
    (and (natp (car x))
         (all-natp (cdr x)))))

(defthm all-natp-nil
  (equal (all-natp nil) t))

(defthm all-natp-cons
  (equal (all-natp (cons a x))
         (and (natp a) (all-natp x))))

(defthm all-natp-implies-true-listp
  (implies (all-natp x)
           (true-listp x))
  :rule-classes (:rewrite :forward-chaining))

(defthm all-natp-cdr
  (implies (and (all-natp x) (consp x))
           (all-natp (cdr x))))

(defthm all-natp-car-natp
  (implies (and (all-natp x) (consp x))
           (natp (car x))))

(defthm all-natp-append
  (implies (and (all-natp x) (all-natp y))
           (all-natp (append x y))))

(defthm all-natp-nthcdr
  (implies (all-natp x)
           (all-natp (nthcdr n x))))

(defthm all-natp-take
  (implies (and (all-natp x)
                (natp n)
                (<= n (len x)))
           (all-natp (take n x))))

(defthm all-natp-revappend
  (implies (and (all-natp x) (all-natp y))
           (all-natp (revappend x y))))

(defthm all-natp-reverse
  (implies (all-natp x)
           (all-natp (reverse x))))

(defun alist-get (key alist)
  (declare (xargs :guard t))
  (let ((pair (assoc-equal key alist)))
    (if pair (cdr pair) nil)))

(defun alist-put (key val alist)
  (declare (xargs :guard t))
  (acons key val alist))

(defun alist-contains (key alist)
  (declare (xargs :guard t))
  (if (assoc-equal key alist) t nil))

(defthm alist-get-of-alist-put-same
  (equal (alist-get k (alist-put k v a))
         v))

(defthm alist-get-of-alist-put-diff
  (implies (not (equal k1 k2))
           (equal (alist-get k1 (alist-put k2 v a))
                  (alist-get k1 a))))

(defthm alist-contains-of-alist-put-same
  (equal (alist-contains k (alist-put k v a))
         t))

(defthm alist-contains-of-alist-put-diff
  (implies (not (equal k1 k2))
           (equal (alist-contains k1 (alist-put k2 v a))
                  (alist-contains k1 a))))

(defthm alist-get-nil
  (equal (alist-get k nil) nil))

(defthm alist-contains-nil
  (equal (alist-contains k nil) nil))

(defthm alistp-alist-put
  (implies (alistp a)
           (alistp (alist-put k v a))))

(defthm consp-assoc-equal-when-alist-contains
  (implies (alist-contains k a)
           (consp (assoc-equal k a))))

(defthm alist-get-type-when-natp-val
  (implies (and (alist-contains k a)
                (natp (cdr (assoc-equal k a))))
           (natp (alist-get k a))))

(defun alist-keys (alist)
  (declare (xargs :guard t))
  (if (atom alist)
      nil
    (if (consp (car alist))
        (cons (caar alist) (alist-keys (cdr alist)))
      (alist-keys (cdr alist)))))

(defthm true-listp-alist-keys
  (true-listp (alist-keys a)))

(defthm member-alist-keys-iff-alist-contains
  (implies (alistp a)
           (iff (member-equal k (alist-keys a))
                (alist-contains k a))))

(defun alist-vals (alist)
  (declare (xargs :guard t))
  (if (atom alist)
      nil
    (if (consp (car alist))
        (cons (cdar alist) (alist-vals (cdr alist)))
      (alist-vals (cdr alist)))))

(defthm true-listp-alist-vals
  (true-listp (alist-vals a)))

(defun alist-count (alist)
  (declare (xargs :guard t))
  (if (atom alist)
      0
    (if (consp (car alist))
        (+ 1 (alist-count (cdr alist)))
      (alist-count (cdr alist)))))

(defthm natp-alist-count
  (natp (alist-count a))
  :rule-classes (:rewrite :type-prescription))

(defthm alist-count-of-alist-put
  (equal (alist-count (alist-put k v a))
         (+ 1 (alist-count a))))

(defconst *pad-id* 0)
(defconst *unk-id* 1)
(defconst *bos-id* 2)
(defconst *eos-id* 3)

(defconst *pad-word* (coerce "[PAD]" 'list))
(defconst *unk-word* (coerce "[UNK]" 'list))
(defconst *bos-word* (coerce "[BOS]" 'list))
(defconst *eos-word* (coerce "[EOS]" 'list))

(defconst *special-tokens*
  (list *pad-word* *unk-word* *bos-word* *eos-word*))

(defconst *special-ids*
  (list *pad-id* *unk-id* *bos-id* *eos-id*))

(defun mgt-make-state (tok2id id2tok prefixes suffixes roots bpe-pairs anchors next-id)
  (declare (xargs :guard t))
  (list tok2id id2tok prefixes suffixes roots bpe-pairs anchors next-id))

(defun mgt-tok2id (st) (declare (xargs :guard t)) (nth 0 st))
(defun mgt-id2tok (st) (declare (xargs :guard t)) (nth 1 st))
(defun mgt-prefixes (st) (declare (xargs :guard t)) (nth 2 st))
(defun mgt-suffixes (st) (declare (xargs :guard t)) (nth 3 st))
(defun mgt-roots (st) (declare (xargs :guard t)) (nth 4 st))
(defun mgt-bpe-pairs (st) (declare (xargs :guard t)) (nth 5 st))
(defun mgt-anchors (st) (declare (xargs :guard t)) (nth 6 st))
(defun mgt-next-id (st) (declare (xargs :guard t)) (nth 7 st))

(defun mgt-set-tok2id (v st)
  (declare (xargs :guard t))
  (update-nth 0 v st))

(defun mgt-set-id2tok (v st)
  (declare (xargs :guard t))
  (update-nth 1 v st))

(defun mgt-set-prefixes (v st)
  (declare (xargs :guard t))
  (update-nth 2 v st))

(defun mgt-set-suffixes (v st)
  (declare (xargs :guard t))
  (update-nth 3 v st))

(defun mgt-set-roots (v st)
  (declare (xargs :guard t))
  (update-nth 4 v st))

(defun mgt-set-bpe-pairs (v st)
  (declare (xargs :guard t))
  (update-nth 5 v st))

(defun mgt-set-anchors (v st)
  (declare (xargs :guard t))
  (update-nth 6 v st))

(defun mgt-set-next-id (v st)
  (declare (xargs :guard t))
  (update-nth 7 v st))

(defthm mgt-tok2id-of-make
  (equal (mgt-tok2id (mgt-make-state a b c d e f g h)) a))

(defthm mgt-id2tok-of-make
  (equal (mgt-id2tok (mgt-make-state a b c d e f g h)) b))

(defthm mgt-prefixes-of-make
  (equal (mgt-prefixes (mgt-make-state a b c d e f g h)) c))

(defthm mgt-suffixes-of-make
  (equal (mgt-suffixes (mgt-make-state a b c d e f g h)) d))

(defthm mgt-roots-of-make
  (equal (mgt-roots (mgt-make-state a b c d e f g h)) e))

(defthm mgt-bpe-pairs-of-make
  (equal (mgt-bpe-pairs (mgt-make-state a b c d e f g h)) f))

(defthm mgt-anchors-of-make
  (equal (mgt-anchors (mgt-make-state a b c d e f g h)) g))

(defthm mgt-next-id-of-make
  (equal (mgt-next-id (mgt-make-state a b c d e f g h)) h))

(defthm mgt-tok2id-of-set-tok2id
  (equal (mgt-tok2id (mgt-set-tok2id v st)) v))

(defthm mgt-id2tok-of-set-id2tok
  (equal (mgt-id2tok (mgt-set-id2tok v st)) v))

(defthm mgt-prefixes-of-set-prefixes
  (equal (mgt-prefixes (mgt-set-prefixes v st)) v))

(defthm mgt-suffixes-of-set-suffixes
  (equal (mgt-suffixes (mgt-set-suffixes v st)) v))

(defthm mgt-roots-of-set-roots
  (equal (mgt-roots (mgt-set-roots v st)) v))

(defthm mgt-bpe-pairs-of-set-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-set-bpe-pairs v st)) v))

(defthm mgt-anchors-of-set-anchors
  (equal (mgt-anchors (mgt-set-anchors v st)) v))

(defthm mgt-next-id-of-set-next-id
  (equal (mgt-next-id (mgt-set-next-id v st)) v))

(defthm mgt-tok2id-of-set-id2tok
  (equal (mgt-tok2id (mgt-set-id2tok v st))
         (mgt-tok2id st)))

(defthm mgt-tok2id-of-set-prefixes
  (equal (mgt-tok2id (mgt-set-prefixes v st))
         (mgt-tok2id st)))

(defthm mgt-tok2id-of-set-suffixes
  (equal (mgt-tok2id (mgt-set-suffixes v st))
         (mgt-tok2id st)))

(defthm mgt-tok2id-of-set-roots
  (equal (mgt-tok2id (mgt-set-roots v st))
         (mgt-tok2id st)))

(defthm mgt-tok2id-of-set-bpe-pairs
  (equal (mgt-tok2id (mgt-set-bpe-pairs v st))
         (mgt-tok2id st)))

(defthm mgt-tok2id-of-set-anchors
  (equal (mgt-tok2id (mgt-set-anchors v st))
         (mgt-tok2id st)))

(defthm mgt-tok2id-of-set-next-id
  (equal (mgt-tok2id (mgt-set-next-id v st))
         (mgt-tok2id st)))

(defthm mgt-id2tok-of-set-tok2id
  (equal (mgt-id2tok (mgt-set-tok2id v st))
         (mgt-id2tok st)))

(defthm mgt-id2tok-of-set-prefixes
  (equal (mgt-id2tok (mgt-set-prefixes v st))
         (mgt-id2tok st)))

(defthm mgt-id2tok-of-set-suffixes
  (equal (mgt-id2tok (mgt-set-suffixes v st))
         (mgt-id2tok st)))

(defthm mgt-id2tok-of-set-roots
  (equal (mgt-id2tok (mgt-set-roots v st))
         (mgt-id2tok st)))

(defthm mgt-id2tok-of-set-bpe-pairs
  (equal (mgt-id2tok (mgt-set-bpe-pairs v st))
         (mgt-id2tok st)))

(defthm mgt-id2tok-of-set-anchors
  (equal (mgt-id2tok (mgt-set-anchors v st))
         (mgt-id2tok st)))

(defthm mgt-id2tok-of-set-next-id
  (equal (mgt-id2tok (mgt-set-next-id v st))
         (mgt-id2tok st)))

(defthm mgt-next-id-of-set-tok2id
  (equal (mgt-next-id (mgt-set-tok2id v st))
         (mgt-next-id st)))

(defthm mgt-next-id-of-set-id2tok
  (equal (mgt-next-id (mgt-set-id2tok v st))
         (mgt-next-id st)))

(defthm mgt-next-id-of-set-prefixes
  (equal (mgt-next-id (mgt-set-prefixes v st))
         (mgt-next-id st)))

(defthm mgt-next-id-of-set-suffixes
  (equal (mgt-next-id (mgt-set-suffixes v st))
         (mgt-next-id st)))

(defthm mgt-next-id-of-set-roots
  (equal (mgt-next-id (mgt-set-roots v st))
         (mgt-next-id st)))

(defthm mgt-next-id-of-set-bpe-pairs
  (equal (mgt-next-id (mgt-set-bpe-pairs v st))
         (mgt-next-id st)))

(defthm mgt-next-id-of-set-anchors
  (equal (mgt-next-id (mgt-set-anchors v st))
         (mgt-next-id st)))

(defthm mgt-prefixes-of-set-tok2id
  (equal (mgt-prefixes (mgt-set-tok2id v st))
         (mgt-prefixes st)))

(defthm mgt-prefixes-of-set-id2tok
  (equal (mgt-prefixes (mgt-set-id2tok v st))
         (mgt-prefixes st)))

(defthm mgt-prefixes-of-set-next-id
  (equal (mgt-prefixes (mgt-set-next-id v st))
         (mgt-prefixes st)))

(defthm mgt-suffixes-of-set-tok2id
  (equal (mgt-suffixes (mgt-set-tok2id v st))
         (mgt-suffixes st)))

(defthm mgt-suffixes-of-set-id2tok
  (equal (mgt-suffixes (mgt-set-id2tok v st))
         (mgt-suffixes st)))

(defthm mgt-suffixes-of-set-next-id
  (equal (mgt-suffixes (mgt-set-next-id v st))
         (mgt-suffixes st)))

(defthm mgt-roots-of-set-tok2id
  (equal (mgt-roots (mgt-set-tok2id v st))
         (mgt-roots st)))

(defthm mgt-roots-of-set-id2tok
  (equal (mgt-roots (mgt-set-id2tok v st))
         (mgt-roots st)))

(defthm mgt-roots-of-set-next-id
  (equal (mgt-roots (mgt-set-next-id v st))
         (mgt-roots st)))

(defthm mgt-anchors-of-set-tok2id
  (equal (mgt-anchors (mgt-set-tok2id v st))
         (mgt-anchors st)))

(defthm mgt-anchors-of-set-id2tok
  (equal (mgt-anchors (mgt-set-id2tok v st))
         (mgt-anchors st)))

(defthm mgt-anchors-of-set-next-id
  (equal (mgt-anchors (mgt-set-next-id v st))
         (mgt-anchors st)))

(defthm mgt-bpe-pairs-of-set-tok2id
  (equal (mgt-bpe-pairs (mgt-set-tok2id v st))
         (mgt-bpe-pairs st)))

(defthm mgt-bpe-pairs-of-set-id2tok
  (equal (mgt-bpe-pairs (mgt-set-id2tok v st))
         (mgt-bpe-pairs st)))

(defthm mgt-bpe-pairs-of-set-next-id
  (equal (mgt-bpe-pairs (mgt-set-next-id v st))
         (mgt-bpe-pairs st)))

(defthm mgt-suffixes-of-set-prefixes
  (equal (mgt-suffixes (mgt-set-prefixes v st))
         (mgt-suffixes st)))

(defthm mgt-prefixes-of-set-suffixes
  (equal (mgt-prefixes (mgt-set-suffixes v st))
         (mgt-prefixes st)))

(defthm mgt-roots-of-set-prefixes
  (equal (mgt-roots (mgt-set-prefixes v st))
         (mgt-roots st)))

(defthm mgt-roots-of-set-suffixes
  (equal (mgt-roots (mgt-set-suffixes v st))
         (mgt-roots st)))

(defthm mgt-prefixes-of-set-roots
  (equal (mgt-prefixes (mgt-set-roots v st))
         (mgt-prefixes st)))

(defthm mgt-suffixes-of-set-roots
  (equal (mgt-suffixes (mgt-set-roots v st))
         (mgt-suffixes st)))

(defthm mgt-bpe-pairs-of-set-prefixes
  (equal (mgt-bpe-pairs (mgt-set-prefixes v st))
         (mgt-bpe-pairs st)))

(defthm mgt-bpe-pairs-of-set-suffixes
  (equal (mgt-bpe-pairs (mgt-set-suffixes v st))
         (mgt-bpe-pairs st)))

(defthm mgt-bpe-pairs-of-set-roots
  (equal (mgt-bpe-pairs (mgt-set-roots v st))
         (mgt-bpe-pairs st)))

(defthm mgt-anchors-of-set-prefixes
  (equal (mgt-anchors (mgt-set-prefixes v st))
         (mgt-anchors st)))

(defthm mgt-anchors-of-set-suffixes
  (equal (mgt-anchors (mgt-set-suffixes v st))
         (mgt-anchors st)))

(defthm mgt-anchors-of-set-roots
  (equal (mgt-anchors (mgt-set-roots v st))
         (mgt-anchors st)))

(defthm mgt-prefixes-of-set-bpe-pairs
  (equal (mgt-prefixes (mgt-set-bpe-pairs v st))
         (mgt-prefixes st)))

(defthm mgt-suffixes-of-set-bpe-pairs
  (equal (mgt-suffixes (mgt-set-bpe-pairs v st))
         (mgt-suffixes st)))

(defthm mgt-roots-of-set-bpe-pairs
  (equal (mgt-roots (mgt-set-bpe-pairs v st))
         (mgt-roots st)))

(defthm mgt-prefixes-of-set-anchors
  (equal (mgt-prefixes (mgt-set-anchors v st))
         (mgt-prefixes st)))

(defthm mgt-suffixes-of-set-anchors
  (equal (mgt-suffixes (mgt-set-anchors v st))
         (mgt-suffixes st)))

(defthm mgt-roots-of-set-anchors
  (equal (mgt-roots (mgt-set-anchors v st))
         (mgt-roots st)))

(defthm mgt-anchors-of-set-bpe-pairs
  (equal (mgt-anchors (mgt-set-bpe-pairs v st))
         (mgt-anchors st)))

(defthm mgt-bpe-pairs-of-set-anchors
  (equal (mgt-bpe-pairs (mgt-set-anchors v st))
         (mgt-bpe-pairs st)))

(defun mgt-statep (st)
  (declare (xargs :guard t))
  (and (true-listp st)
       (equal (len st) 8)
       (alistp (mgt-tok2id st))
       (alistp (mgt-id2tok st))
       (alistp (mgt-prefixes st))
       (alistp (mgt-suffixes st))
       (alistp (mgt-roots st))
       (alistp (mgt-bpe-pairs st))
       (alistp (mgt-anchors st))
       (natp (mgt-next-id st))))

(defthm mgt-statep-of-make
  (implies (and (alistp a) (alistp b) (alistp c) (alistp d)
                (alistp e) (alistp f) (alistp g) (natp h))
           (mgt-statep (mgt-make-state a b c d e f g h))))

(defthm mgt-statep-implies-true-listp
  (implies (mgt-statep st)
           (true-listp st))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-natp-next-id
  (implies (mgt-statep st)
           (natp (mgt-next-id st)))
  :rule-classes (:rewrite :forward-chaining :type-prescription))

(defthm mgt-statep-implies-alistp-tok2id
  (implies (mgt-statep st)
           (alistp (mgt-tok2id st)))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-alistp-id2tok
  (implies (mgt-statep st)
           (alistp (mgt-id2tok st)))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-alistp-prefixes
  (implies (mgt-statep st)
           (alistp (mgt-prefixes st)))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-alistp-suffixes
  (implies (mgt-statep st)
           (alistp (mgt-suffixes st)))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-alistp-roots
  (implies (mgt-statep st)
           (alistp (mgt-roots st)))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-alistp-bpe-pairs
  (implies (mgt-statep st)
           (alistp (mgt-bpe-pairs st)))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-alistp-anchors
  (implies (mgt-statep st)
           (alistp (mgt-anchors st)))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-statep-implies-len-8
  (implies (mgt-statep st)
           (equal (len st) 8)))

(defun mgt-add-token (word st)
  (declare (xargs :guard t))
  (if (alist-contains word (mgt-tok2id st))
      (list (alist-get word (mgt-tok2id st)) st)
    (let* ((id (nfix (mgt-next-id st)))
           (new-tok2id (alist-put word id (mgt-tok2id st)))
           (new-id2tok (alist-put id word (mgt-id2tok st)))
           (st1 (mgt-set-tok2id new-tok2id st))
           (st2 (mgt-set-id2tok new-id2tok st1))
           (st3 (mgt-set-next-id (+ 1 id) st2)))
      (list id st3))))

(defun mgt-add-token-id (word st)
  (declare (xargs :guard t))
  (car (mgt-add-token word st)))

(defun mgt-add-token-state (word st)
  (declare (xargs :guard t))
  (cadr (mgt-add-token word st)))

(defthm natp-mgt-add-token-id
  (implies (mgt-statep st)
           (natp (mgt-add-token-id word st)))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-add-token-id-existing
  (implies (alist-contains word (mgt-tok2id st))
           (equal (mgt-add-token-id word st)
                  (alist-get word (mgt-tok2id st)))))

(defthm mgt-add-token-state-existing
  (implies (alist-contains word (mgt-tok2id st))
           (equal (mgt-add-token-state word st)
                  st)))

(defthm mgt-add-token-id-new
  (implies (and (mgt-statep st)
                (not (alist-contains word (mgt-tok2id st))))
           (equal (mgt-add-token-id word st)
                  (nfix (mgt-next-id st)))))

(defthm mgt-add-token-next-id-increases
  (implies (and (mgt-statep st)
                (not (alist-contains word (mgt-tok2id st))))
           (equal (mgt-next-id (mgt-add-token-state word st))
                  (+ 1 (nfix (mgt-next-id st))))))

(defthm mgt-add-token-findable
  (implies (mgt-statep st)
           (alist-contains word (mgt-tok2id (mgt-add-token-state word st)))))

(defthm mgt-add-token-id-retrievable
  (implies (mgt-statep st)
           (equal (alist-get word (mgt-tok2id (mgt-add-token-state word st)))
                  (mgt-add-token-id word st))))

(defthm mgt-add-token-reverse-map
  (implies (and (mgt-statep st)
                (not (alist-contains word (mgt-tok2id st))))
           (equal (alist-get (nfix (mgt-next-id st))
                             (mgt-id2tok (mgt-add-token-state word st)))
                  word)))

(defthm mgt-add-token-preserves-prefixes
  (equal (mgt-prefixes (mgt-add-token-state word st))
         (mgt-prefixes st)))

(defthm mgt-add-token-preserves-suffixes
  (equal (mgt-suffixes (mgt-add-token-state word st))
         (mgt-suffixes st)))

(defthm mgt-add-token-preserves-roots
  (equal (mgt-roots (mgt-add-token-state word st))
         (mgt-roots st)))

(defthm mgt-add-token-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-add-token-state word st))
         (mgt-bpe-pairs st)))

(defthm mgt-add-token-preserves-anchors
  (equal (mgt-anchors (mgt-add-token-state word st))
         (mgt-anchors st)))

(defthm mgt-add-token-preserves-existing
  (implies (and (not (equal w1 w2))
                (alist-contains w1 (mgt-tok2id st)))
           (alist-contains w1 (mgt-tok2id (mgt-add-token-state w2 st)))))

(defthm mgt-add-token-preserves-existing-id
  (implies (and (not (equal w1 w2))
                (alist-contains w1 (mgt-tok2id st)))
           (equal (alist-get w1 (mgt-tok2id (mgt-add-token-state w2 st)))
                  (alist-get w1 (mgt-tok2id st)))))

(defun mgt-add-tokens (words st)
  (declare (xargs :guard t
                  :measure (acl2-count words)))
  (if (atom words)
      st
    (let ((st2 (mgt-add-token-state (car words) st)))
      (mgt-add-tokens (cdr words) st2))))

(defthm mgt-add-tokens-nil
  (equal (mgt-add-tokens nil st) st))

(defthm mgt-add-tokens-cons
  (equal (mgt-add-tokens (cons w ws) st)
         (mgt-add-tokens ws (mgt-add-token-state w st))))

(defthm mgt-add-tokens-preserves-prefixes
  (equal (mgt-prefixes (mgt-add-tokens words st))
         (mgt-prefixes st))
  :hints (("Goal" :induct (mgt-add-tokens words st))))

(defthm mgt-add-tokens-preserves-suffixes
  (equal (mgt-suffixes (mgt-add-tokens words st))
         (mgt-suffixes st))
  :hints (("Goal" :induct (mgt-add-tokens words st))))

(defthm mgt-add-tokens-preserves-roots
  (equal (mgt-roots (mgt-add-tokens words st))
         (mgt-roots st))
  :hints (("Goal" :induct (mgt-add-tokens words st))))

(defthm mgt-add-tokens-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-add-tokens words st))
         (mgt-bpe-pairs st))
  :hints (("Goal" :induct (mgt-add-tokens words st))))

(defthm mgt-add-tokens-preserves-anchors
  (equal (mgt-anchors (mgt-add-tokens words st))
         (mgt-anchors st))
  :hints (("Goal" :induct (mgt-add-tokens words st))))

(defun mgt-empty-state ()
  (declare (xargs :guard t))
  (mgt-make-state nil nil nil nil nil nil nil 0))

(defthm mgt-statep-empty
  (mgt-statep (mgt-empty-state)))

(defthm mgt-next-id-empty
  (equal (mgt-next-id (mgt-empty-state)) 0))

(defthm mgt-tok2id-empty
  (equal (mgt-tok2id (mgt-empty-state)) nil))

(defthm mgt-id2tok-empty
  (equal (mgt-id2tok (mgt-empty-state)) nil))

(defun mgt-vocab-size (st)
  (declare (xargs :guard t))
  (alist-count (mgt-tok2id st)))

(defthm natp-mgt-vocab-size
  (natp (mgt-vocab-size st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-vocab-size-empty
  (equal (mgt-vocab-size (mgt-empty-state)) 0))

(defthm mgt-vocab-size-after-add-new
  (implies (and (mgt-statep st)
                (not (alist-contains word (mgt-tok2id st))))
           (equal (mgt-vocab-size (mgt-add-token-state word st))
                  (+ 1 (mgt-vocab-size st)))))

(defthm mgt-vocab-size-after-add-existing
  (implies (alist-contains word (mgt-tok2id st))
           (equal (mgt-vocab-size (mgt-add-token-state word st))
                  (mgt-vocab-size st))))

(defun whitespace-byte-p (b)
  (declare (xargs :guard t))
  (or (equal b #\Space)
      (equal b #\Newline)
      (equal b #\Tab)
      (equal b #\Return)))

(defun punctuation-byte-p (b)
  (declare (xargs :guard t))
  (or (equal b #\.)
      (equal b #\,)
      (equal b #\!)
      (equal b #\?)
      (equal b #\;)
      (equal b #\:)
      (equal b #\")
      (equal b #\')
      (equal b #\()
      (equal b #\))
      (equal b #\{)
      (equal b #\})))

(defthm booleanp-whitespace-byte-p
  (or (equal (whitespace-byte-p b) t)
      (equal (whitespace-byte-p b) nil))
  :rule-classes :type-prescription)

(defthm booleanp-punctuation-byte-p
  (or (equal (punctuation-byte-p b) t)
      (equal (punctuation-byte-p b) nil))
  :rule-classes :type-prescription)

(defthm whitespace-not-punctuation
  (implies (whitespace-byte-p b)
           (not (punctuation-byte-p b))))

(defthm space-is-whitespace
  (equal (whitespace-byte-p #\Space) t))

(defthm newline-is-whitespace
  (equal (whitespace-byte-p #\Newline) t))

(defthm tab-is-whitespace
  (equal (whitespace-byte-p #\Tab) t))

(defthm return-is-whitespace
  (equal (whitespace-byte-p #\Return) t))

(defthm period-is-punctuation
  (equal (punctuation-byte-p #\.) t))

(defthm comma-is-punctuation
  (equal (punctuation-byte-p #\,) t))

(defthm exclam-is-punctuation
  (equal (punctuation-byte-p #\!) t))

(defthm question-is-punctuation
  (equal (punctuation-byte-p #\?) t))

(defun starts-with-p (text prefix)
  (declare (xargs :guard t
                  :measure (acl2-count prefix)))
  (cond ((atom prefix) t)
        ((atom text) nil)
        ((equal (car text) (car prefix))
         (starts-with-p (cdr text) (cdr prefix)))
        (t nil)))

(defthm starts-with-p-nil-prefix
  (equal (starts-with-p text nil) t))

(defthm starts-with-p-nil-text
  (implies (consp prefix)
           (equal (starts-with-p nil prefix) nil)))

(defthm starts-with-p-reflexive
  (starts-with-p text text))

(defthm starts-with-p-cons-cons
  (equal (starts-with-p (cons a x) (cons b y))
         (and (equal a b) (starts-with-p x y))))

(defthm starts-with-p-append
  (implies (true-listp prefix)
           (starts-with-p (append prefix rest) prefix))
  :hints (("Goal" :induct (starts-with-p (append prefix rest) prefix))))

(defthm starts-with-p-implies-len
  (implies (and (starts-with-p text prefix)
                (true-listp prefix)
                (true-listp text))
           (<= (len prefix) (len text)))
  :rule-classes :linear)

(defun special-token-at (text)
  (declare (xargs :guard t))
  (cond ((starts-with-p text *pad-word*) (len *pad-word*))
        ((starts-with-p text *unk-word*) (len *unk-word*))
        ((starts-with-p text *bos-word*) (len *bos-word*))
        ((starts-with-p text *eos-word*) (len *eos-word*))
        (t 0)))

(defthm natp-special-token-at
  (natp (special-token-at text))
  :rule-classes (:rewrite :type-prescription))

(defthm special-token-at-nil
  (equal (special-token-at nil) 0))

(defthm special-token-at-pad
  (equal (special-token-at *pad-word*) (len *pad-word*)))

(defthm special-token-at-unk
  (equal (special-token-at *unk-word*) (len *unk-word*)))

(defthm special-token-at-bos
  (equal (special-token-at *bos-word*) (len *bos-word*)))

(defthm special-token-at-eos
  (equal (special-token-at *eos-word*) (len *eos-word*)))

(defthm special-token-at-positive-implies-starts-with-special
  (implies (> (special-token-at text) 0)
           (or (starts-with-p text *pad-word*)
               (starts-with-p text *unk-word*)
               (starts-with-p text *bos-word*)
               (starts-with-p text *eos-word*))))

(defun get-special-token-word (text)
  (declare (xargs :guard t))
  (cond ((starts-with-p text *pad-word*) *pad-word*)
        ((starts-with-p text *unk-word*) *unk-word*)
        ((starts-with-p text *bos-word*) *bos-word*)
        ((starts-with-p text *eos-word*) *eos-word*)
        (t nil)))

(defthm get-special-token-word-nil-when-zero
  (implies (equal (special-token-at text) 0)
           (equal (get-special-token-word text) nil)))

(defthm get-special-token-word-non-nil-when-positive
  (implies (> (special-token-at text) 0)
           (get-special-token-word text)))

(defthm len-get-special-token-word
  (implies (> (special-token-at text) 0)
           (equal (len (get-special-token-word text))
                  (special-token-at text))))

(defun extract-word-chars (text)
  (declare (xargs :guard t
                  :measure (acl2-count text)))
  (cond ((atom text) nil)
        ((whitespace-byte-p (car text)) nil)
        ((punctuation-byte-p (car text)) nil)
        (t (cons (car text) (extract-word-chars (cdr text))))))

(defthm true-listp-extract-word-chars
  (true-listp (extract-word-chars text))
  :rule-classes (:rewrite :type-prescription))

(defthm extract-word-chars-nil
  (equal (extract-word-chars nil) nil))

(defthm extract-word-chars-whitespace
  (implies (whitespace-byte-p (car text))
           (equal (extract-word-chars text) nil)))

(defthm extract-word-chars-punctuation
  (implies (punctuation-byte-p (car text))
           (equal (extract-word-chars text) nil)))

(defthm len-extract-word-chars-le
  (<= (len (extract-word-chars text)) (len text))
  :rule-classes :linear)

(defthm extract-word-chars-non-nil-when-word-start
  (implies (and (consp text)
                (not (whitespace-byte-p (car text)))
                (not (punctuation-byte-p (car text))))
           (consp (extract-word-chars text))))

(defthm len-extract-word-chars-positive-when-word-start
  (implies (and (consp text)
                (not (whitespace-byte-p (car text)))
                (not (punctuation-byte-p (car text))))
           (> (len (extract-word-chars text)) 0))
  :rule-classes :linear)

(defun longest-match-aux (text len max-len best tok2id)
  (declare (xargs :guard (and (natp len) (natp max-len) (natp best))
                  :measure (nfix (- (nfix max-len) (nfix len)))))
  (if (or (not (natp len))
          (not (natp max-len))
          (> len max-len)
          (zp (- max-len len)))
      best
    (let ((candidate (take (+ 1 len) text)))
      (if (alist-contains candidate tok2id)
          (longest-match-aux text (+ 1 len) max-len (+ 1 len) tok2id)
        (longest-match-aux text (+ 1 len) max-len best tok2id)))))

(defthm natp-longest-match-aux
  (implies (natp best)
           (natp (longest-match-aux text len max-len best tok2id)))
  :rule-classes (:rewrite :type-prescription))

(defthm longest-match-aux-ge-best
  (implies (natp best)
           (<= best (longest-match-aux text len max-len best tok2id)))
  :rule-classes :linear)

(defthm longest-match-aux-le-max-len
  (implies (and (natp best)
                (natp max-len)
                (<= best max-len))
           (<= (longest-match-aux text len max-len best tok2id) max-len))
  :rule-classes :linear
  :hints (("Goal" :induct (longest-match-aux text len max-len best tok2id))))

(defun mgt-longest-match (text st)
  (declare (xargs :guard t))
  (if (atom text)
      0
    (longest-match-aux text 0 (len text) 0 (mgt-tok2id st))))

(defthm natp-mgt-longest-match
  (natp (mgt-longest-match text st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-longest-match-nil
  (equal (mgt-longest-match nil st) 0))

(defthm mgt-longest-match-le-len
  (<= (mgt-longest-match text st) (len text))
  :rule-classes :linear)

(defthm mgt-longest-match-zero-empty-vocab
  (equal (mgt-longest-match text (mgt-empty-state)) 0)
  :hints (("Goal" :expand ((mgt-longest-match text (mgt-empty-state))))))

(defun find-longest-prefix-aux (word len max-check prefixes best-len)
  (declare (xargs :guard (and (natp len) (natp max-check) (natp best-len))
                  :measure (nfix (- (nfix max-check) (nfix len)))))
  (if (or (not (natp len))
          (not (natp max-check))
          (>= len max-check)
          (zp (- max-check len)))
      best-len
    (let ((candidate (take (+ 1 len) word)))
      (if (alist-contains candidate prefixes)
          (find-longest-prefix-aux word (+ 1 len) max-check prefixes (+ 1 len))
        (find-longest-prefix-aux word (+ 1 len) max-check prefixes best-len)))))

(defthm natp-find-longest-prefix-aux
  (implies (natp best-len)
           (natp (find-longest-prefix-aux word len max-check prefixes best-len)))
  :rule-classes (:rewrite :type-prescription))

(defthm find-longest-prefix-aux-ge-best
  (implies (natp best-len)
           (<= best-len (find-longest-prefix-aux word len max-check prefixes best-len)))
  :rule-classes :linear)

(defun mgt-find-longest-prefix (word st)
  (declare (xargs :guard t))
  (if (or (atom word) (atom (cdr word)))
      0
    (find-longest-prefix-aux word 0 (- (len word) 1) (mgt-prefixes st) 0)))

(defthm natp-mgt-find-longest-prefix
  (natp (mgt-find-longest-prefix word st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-find-longest-prefix-nil
  (equal (mgt-find-longest-prefix nil st) 0))

(defthm mgt-find-longest-prefix-singleton
  (implies (and (consp word) (atom (cdr word)))
           (equal (mgt-find-longest-prefix word st) 0)))

(defun find-longest-suffix-aux (word len max-check suffixes best-len)
  (declare (xargs :guard (and (natp len) (natp max-check) (natp best-len) (true-listp word))
                  :measure (nfix (- (nfix max-check) (nfix len)))))
  (if (or (not (natp len))
          (not (natp max-check))
          (>= len max-check)
          (zp (- max-check len)))
      best-len
    (let ((candidate (nthcdr (- (len word) (+ 1 len)) word)))
      (if (alist-contains candidate suffixes)
          (find-longest-suffix-aux word (+ 1 len) max-check suffixes (+ 1 len))
        (find-longest-suffix-aux word (+ 1 len) max-check suffixes best-len)))))

(defthm natp-find-longest-suffix-aux
  (implies (natp best-len)
           (natp (find-longest-suffix-aux word len max-check suffixes best-len)))
  :rule-classes (:rewrite :type-prescription))

(defthm find-longest-suffix-aux-ge-best
  (implies (natp best-len)
           (<= best-len (find-longest-suffix-aux word len max-check suffixes best-len)))
  :rule-classes :linear)

(defun mgt-find-longest-suffix (word st)
  (declare (xargs :guard t))
  (if (or (atom word) (atom (cdr word)) (not (true-listp word)))
      0
    (find-longest-suffix-aux word 0 (- (len word) 1) (mgt-suffixes st) 0)))

(defthm natp-mgt-find-longest-suffix
  (natp (mgt-find-longest-suffix word st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-find-longest-suffix-nil
  (equal (mgt-find-longest-suffix nil st) 0))

(defthm mgt-find-longest-suffix-singleton
  (implies (and (consp word) (atom (cdr word)))
           (equal (mgt-find-longest-suffix word st) 0)))

(defun mgt-morph-decompose (word st)
  (declare (xargs :guard t))
  (if (or (atom word) (< (len word) 4) (not (true-listp word)))
      nil
    (let* ((prefix-len (mgt-find-longest-prefix word st))
           (suffix-len (mgt-find-longest-suffix word st)))
      (if (and (equal prefix-len 0) (equal suffix-len 0))
          nil
        (let* ((root-start prefix-len)
               (root-end (- (len word) suffix-len)))
          (if (or (<= root-end root-start)
                  (< (- root-end root-start) 2))
              nil
            (let* ((prefix-word (if (> prefix-len 0) (take prefix-len word) nil))
                   (root-word (take (- root-end root-start) (nthcdr root-start word)))
                   (suffix-word (if (> suffix-len 0) (nthcdr (- (len word) suffix-len) word) nil))
                   (tok2id (mgt-tok2id st))
                   (roots-map (mgt-roots st))
                   (prefix-id (if prefix-word (alist-get prefix-word tok2id) nil))
                   (root-id (or (alist-get root-word tok2id)
                                (alist-get root-word roots-map)))
                   (suffix-id (if suffix-word (alist-get suffix-word tok2id) nil)))
              (if (and (or (null prefix-word) prefix-id)
                       root-id
                       (or (null suffix-word) suffix-id))
                  (append (if prefix-id (list prefix-id) nil)
                          (list root-id)
                          (if suffix-id (list suffix-id) nil))
                nil))))))))

(defthm true-listp-mgt-morph-decompose
  (true-listp (mgt-morph-decompose word st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-morph-decompose-nil
  (equal (mgt-morph-decompose nil st) nil))

(defthm mgt-morph-decompose-short-word
  (implies (< (len word) 4)
           (equal (mgt-morph-decompose word st) nil)))

(defthm mgt-morph-decompose-no-affixes
  (implies (and (equal (mgt-find-longest-prefix word st) 0)
                (equal (mgt-find-longest-suffix word st) 0)
                (true-listp word)
                (>= (len word) 4))
           (equal (mgt-morph-decompose word st) nil)))

(defthm len-mgt-morph-decompose-le-3
  (implies (mgt-morph-decompose word st)
           (<= (len (mgt-morph-decompose word st)) 3))
  :rule-classes :linear)

(defthm len-mgt-morph-decompose-ge-1
  (implies (mgt-morph-decompose word st)
           (>= (len (mgt-morph-decompose word st)) 1))
  :rule-classes :linear)

(defun mgt-encode-bpe-byte (b st)
  (declare (xargs :guard t))
  (let ((byte-word (list b)))
    (let ((tid (alist-get byte-word (mgt-tok2id st))))
      (if tid
          (list tid)
        (list *unk-id*)))))

(defthm true-listp-mgt-encode-bpe-byte
  (true-listp (mgt-encode-bpe-byte b st))
  :rule-classes (:rewrite :type-prescription))

(defthm consp-mgt-encode-bpe-byte
  (consp (mgt-encode-bpe-byte b st)))

(defthm len-mgt-encode-bpe-byte
  (equal (len (mgt-encode-bpe-byte b st)) 1))

(defun mgt-subword-split (word st)
  (declare (xargs :guard t
                  :measure (acl2-count word)))
  (if (atom word)
      nil
    (let ((match-len (mgt-longest-match word st)))
      (if (> match-len 0)
          (let ((matched (take match-len word))
                (rest (nthcdr match-len word)))
            (let ((tid (alist-get matched (mgt-tok2id st))))
              (if tid
                  (cons tid (mgt-subword-split rest st))
                (append (mgt-encode-bpe-byte (car word) st)
                        (mgt-subword-split (cdr word) st)))))
        (append (mgt-encode-bpe-byte (car word) st)
                (mgt-subword-split (cdr word) st))))))

(defthm true-listp-mgt-subword-split
  (true-listp (mgt-subword-split word st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-subword-split-nil
  (equal (mgt-subword-split nil st) nil))

(defthm mgt-subword-split-produces-list
  (implies (consp word)
           (consp (mgt-subword-split word st))))

(defun mgt-encode-word (word st)
  (declare (xargs :guard t))
  (if (atom word)
      nil
    (let ((tid (alist-get word (mgt-tok2id st))))
      (if tid
          (list tid)
        (let ((morph (mgt-morph-decompose word st)))
          (if morph
              morph
            (mgt-subword-split word st)))))))

(defthm true-listp-mgt-encode-word
  (true-listp (mgt-encode-word word st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-encode-word-nil
  (equal (mgt-encode-word nil st) nil))

(defthm mgt-encode-word-known
  (implies (alist-contains word (mgt-tok2id st))
           (equal (mgt-encode-word word st)
                  (list (alist-get word (mgt-tok2id st))))))

(defthm consp-mgt-encode-word-when-consp
  (implies (consp word)
           (consp (mgt-encode-word word st))))

(defthm len-mgt-encode-word-known-is-1
  (implies (alist-contains word (mgt-tok2id st))
           (equal (len (mgt-encode-word word st)) 1)))

(defun mgt-encode (text st)
  (declare (xargs :guard t
                  :measure (acl2-count text)))
  (if (atom text)
      nil
    (let ((slen (special-token-at text)))
      (if (> slen 0)
          (let* ((special-word (get-special-token-word text))
                 (tid (alist-get special-word (mgt-tok2id st)))
                 (rest (nthcdr slen text)))
            (cons (if tid tid *unk-id*)
                  (mgt-encode rest st)))
        (if (whitespace-byte-p (car text))
            (let* ((ws-word (list (car text)))
                   (tid (alist-get ws-word (mgt-tok2id st))))
              (cons (if tid tid *unk-id*)
                    (mgt-encode (cdr text) st)))
          (if (punctuation-byte-p (car text))
              (let* ((p-word (list (car text)))
                     (tid (alist-get p-word (mgt-tok2id st))))
                (cons (if tid tid *unk-id*)
                      (mgt-encode (cdr text) st)))
            (let* ((word-chars (extract-word-chars text))
                   (word-len (len word-chars))
                   (rest (nthcdr word-len text)))
              (if (> word-len 0)
                  (append (mgt-encode-word word-chars st)
                          (mgt-encode rest st))
                (cons *unk-id* (mgt-encode (cdr text) st))))))))))

(defthm true-listp-mgt-encode
  (true-listp (mgt-encode text st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-encode-nil
  (equal (mgt-encode nil st) nil))

(defthm mgt-encode-cons-whitespace
  (implies (and (whitespace-byte-p (car text))
                (consp text)
                (equal (special-token-at text) 0))
           (equal (mgt-encode text st)
                  (cons (let ((tid (alist-get (list (car text)) (mgt-tok2id st))))
                          (if tid tid *unk-id*))
                        (mgt-encode (cdr text) st)))))

(defthm mgt-encode-cons-punctuation
  (implies (and (punctuation-byte-p (car text))
                (not (whitespace-byte-p (car text)))
                (consp text)
                (equal (special-token-at text) 0))
           (equal (mgt-encode text st)
                  (cons (let ((tid (alist-get (list (car text)) (mgt-tok2id st))))
                          (if tid tid *unk-id*))
                        (mgt-encode (cdr text) st)))))

(defun mgt-decode (tokens st)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      nil
    (let* ((tid (car tokens))
           (rest (mgt-decode (cdr tokens) st)))
      (cond ((equal tid *pad-id*) rest)
            ((equal tid *bos-id*) rest)
            ((equal tid *eos-id*) rest)
            ((equal tid *unk-id*) (append *unk-word* rest))
            (t (let ((tok-str (alist-get tid (mgt-id2tok st))))
                 (if tok-str
                     (append tok-str rest)
                   (append *unk-word* rest))))))))

(defthm true-listp-mgt-decode
  (true-listp (mgt-decode tokens st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-decode-nil
  (equal (mgt-decode nil st) nil))

(defthm mgt-decode-pad
  (equal (mgt-decode (list *pad-id*) st) nil))

(defthm mgt-decode-bos
  (equal (mgt-decode (list *bos-id*) st) nil))

(defthm mgt-decode-eos
  (equal (mgt-decode (list *eos-id*) st) nil))

(defthm mgt-decode-unk
  (equal (mgt-decode (list *unk-id*) st) *unk-word*))

(defthm mgt-decode-cons
  (equal (mgt-decode (cons tok rest) st)
         (let ((decoded-rest (mgt-decode rest st)))
           (cond ((equal tok *pad-id*) decoded-rest)
                 ((equal tok *bos-id*) decoded-rest)
                 ((equal tok *eos-id*) decoded-rest)
                 ((equal tok *unk-id*) (append *unk-word* decoded-rest))
                 (t (let ((tok-str (alist-get tok (mgt-id2tok st))))
                      (if tok-str
                          (append tok-str decoded-rest)
                        (append *unk-word* decoded-rest))))))))

(defthm mgt-decode-append
  (equal (mgt-decode (append toks1 toks2) st)
         (append (mgt-decode toks1 st)
                 (mgt-decode toks2 st)))
  :hints (("Goal" :induct (mgt-decode toks1 st))))

(defthm mgt-decode-single-known
  (implies (and (alist-contains tid (mgt-id2tok st))
                (not (equal tid *pad-id*))
                (not (equal tid *bos-id*))
                (not (equal tid *eos-id*))
                (not (equal tid *unk-id*)))
           (equal (mgt-decode (list tid) st)
                  (alist-get tid (mgt-id2tok st)))))

(defun mgt-validate-tokens (tokens st)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      t
    (and (alist-contains (car tokens) (mgt-id2tok st))
         (mgt-validate-tokens (cdr tokens) st))))

(defthm booleanp-mgt-validate-tokens
  (or (equal (mgt-validate-tokens tokens st) t)
      (equal (mgt-validate-tokens tokens st) nil))
  :rule-classes :type-prescription)

(defthm mgt-validate-tokens-nil
  (equal (mgt-validate-tokens nil st) t))

(defthm mgt-validate-tokens-cons
  (equal (mgt-validate-tokens (cons tok rest) st)
         (and (alist-contains tok (mgt-id2tok st))
              (mgt-validate-tokens rest st))))

(defthm mgt-validate-tokens-append
  (equal (mgt-validate-tokens (append a b) st)
         (and (mgt-validate-tokens a st)
              (mgt-validate-tokens b st)))
  :hints (("Goal" :induct (mgt-validate-tokens a st))))

(defthm mgt-validate-tokens-cdr
  (implies (mgt-validate-tokens tokens st)
           (mgt-validate-tokens (cdr tokens) st)))

(defthm mgt-validate-single
  (equal (mgt-validate-tokens (list tok) st)
         (if (alist-contains tok (mgt-id2tok st)) t nil)))

(defun mgt-coverage-aux (text covered st)
  (declare (xargs :guard (natp covered)
                  :measure (acl2-count text)))
  (if (atom text)
      covered
    (let ((m (mgt-longest-match text st)))
      (if (> m 0)
          (mgt-coverage-aux (nthcdr m text) (+ covered m) st)
        (mgt-coverage-aux (cdr text) covered st)))))

(defthm natp-mgt-coverage-aux
  (implies (natp covered)
           (natp (mgt-coverage-aux text covered st)))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-coverage-aux-ge-covered
  (implies (natp covered)
           (<= covered (mgt-coverage-aux text covered st)))
  :rule-classes :linear)

(defthm mgt-coverage-aux-nil
  (implies (natp covered)
           (equal (mgt-coverage-aux nil covered st) covered)))

(defun mgt-coverage-count (text st)
  (declare (xargs :guard t))
  (mgt-coverage-aux text 0 st))

(defthm natp-mgt-coverage-count
  (natp (mgt-coverage-count text st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-coverage-count-nil
  (equal (mgt-coverage-count nil st) 0))

(defthm mgt-coverage-count-le-len
  (<= (mgt-coverage-count text st) (len text))
  :rule-classes :linear
  :hints (("Goal" :in-theory (enable mgt-coverage-count)
           :induct (mgt-coverage-aux text 0 st))))

(defun mgt-coverage (text st)
  (declare (xargs :guard t))
  (if (atom text)
      0
    (let ((total (len text))
          (covered (mgt-coverage-count text st)))
      (if (equal total 0)
          0
        (/ covered total)))))

(defthm rationalp-mgt-coverage
  (rationalp (mgt-coverage text st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-coverage-nil
  (equal (mgt-coverage nil st) 0))

(defthm mgt-coverage-non-negative
  (<= 0 (mgt-coverage text st))
  :rule-classes :linear)

(defthm mgt-coverage-le-1
  (<= (mgt-coverage text st) 1)
  :rule-classes :linear)

(defthm mgt-coverage-zero-empty-vocab
  (equal (mgt-coverage text (mgt-empty-state)) 0)
  :hints (("Goal" :in-theory (enable mgt-coverage mgt-coverage-count))))

(defun mgt-init-special-tokens (st)
  (declare (xargs :guard t))
  (let* ((st1 (mgt-add-token-state *pad-word* st))
         (st2 (mgt-add-token-state *unk-word* st1))
         (st3 (mgt-add-token-state *bos-word* st2))
         (st4 (mgt-add-token-state *eos-word* st3)))
    st4))

(defthm mgt-init-special-tokens-has-pad
  (alist-contains *pad-word*
                  (mgt-tok2id (mgt-init-special-tokens st))))

(defthm mgt-init-special-tokens-has-unk
  (alist-contains *unk-word*
                  (mgt-tok2id (mgt-init-special-tokens st))))

(defthm mgt-init-special-tokens-has-bos
  (alist-contains *bos-word*
                  (mgt-tok2id (mgt-init-special-tokens st))))

(defthm mgt-init-special-tokens-has-eos
  (alist-contains *eos-word*
                  (mgt-tok2id (mgt-init-special-tokens st))))

(defthm mgt-init-special-tokens-preserves-prefixes
  (equal (mgt-prefixes (mgt-init-special-tokens st))
         (mgt-prefixes st)))

(defthm mgt-init-special-tokens-preserves-suffixes
  (equal (mgt-suffixes (mgt-init-special-tokens st))
         (mgt-suffixes st)))

(defthm mgt-init-special-tokens-preserves-roots
  (equal (mgt-roots (mgt-init-special-tokens st))
         (mgt-roots st)))

(defthm mgt-init-special-tokens-preserves-anchors
  (equal (mgt-anchors (mgt-init-special-tokens st))
         (mgt-anchors st)))

(defthm mgt-init-special-tokens-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-init-special-tokens st))
         (mgt-bpe-pairs st)))

(defun mgt-init (vocab-words)
  (declare (xargs :guard t
                  :measure (acl2-count vocab-words)))
  (let* ((st0 (mgt-empty-state))
         (st1 (mgt-init-special-tokens st0))
         (st2 (mgt-add-tokens vocab-words st1)))
    st2))

(defthm mgt-init-has-special-pad
  (alist-contains *pad-word* (mgt-tok2id (mgt-init vocab))))

(defthm mgt-init-has-special-unk
  (alist-contains *unk-word* (mgt-tok2id (mgt-init vocab))))

(defthm mgt-init-has-special-bos
  (alist-contains *bos-word* (mgt-tok2id (mgt-init vocab))))

(defthm mgt-init-has-special-eos
  (alist-contains *eos-word* (mgt-tok2id (mgt-init vocab))))

(defthm mgt-init-nil-vocab-size-ge-4
  (<= 4 (mgt-vocab-size (mgt-init nil)))
  :rule-classes :linear)

(defun mgt-add-vocab-word (word is-anchor st)
  (declare (xargs :guard t))
  (let* ((result (mgt-add-token word st))
         (tid (car result))
         (st2 (cadr result)))
    (if is-anchor
        (mgt-set-anchors (alist-put word tid (mgt-anchors st2)) st2)
      st2)))

(defthm mgt-add-vocab-word-has-word
  (alist-contains word (mgt-tok2id (mgt-add-vocab-word word is-anchor st))))

(defthm mgt-add-vocab-word-anchor-present
  (implies is-anchor
           (alist-contains word (mgt-anchors (mgt-add-vocab-word word t st)))))

(defthm mgt-add-vocab-word-non-anchor-preserves-anchors
  (implies (not is-anchor)
           (equal (mgt-anchors (mgt-add-vocab-word word nil st))
                  (mgt-anchors st))))

(defthm mgt-add-vocab-word-preserves-prefixes
  (equal (mgt-prefixes (mgt-add-vocab-word word is-anchor st))
         (mgt-prefixes st)))

(defthm mgt-add-vocab-word-preserves-suffixes
  (equal (mgt-suffixes (mgt-add-vocab-word word is-anchor st))
         (mgt-suffixes st)))

(defthm mgt-add-vocab-word-preserves-roots
  (equal (mgt-roots (mgt-add-vocab-word word is-anchor st))
         (mgt-roots st)))

(defthm mgt-add-vocab-word-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-add-vocab-word word is-anchor st))
         (mgt-bpe-pairs st)))

(defun mgt-remove-vocab-word (word st)
  (declare (xargs :guard t))
  (if (or (equal word *pad-word*)
          (equal word *unk-word*)
          (equal word *bos-word*)
          (equal word *eos-word*))
      st
    (let ((tid (alist-get word (mgt-tok2id st))))
      (if (not tid)
          st
        (let* ((new-tok2id (remove-assoc-equal word (mgt-tok2id st)))
               (new-id2tok (remove-assoc-equal tid (mgt-id2tok st)))
               (new-anchors (remove-assoc-equal word (mgt-anchors st)))
               (new-prefixes (remove-assoc-equal word (mgt-prefixes st)))
               (new-suffixes (remove-assoc-equal word (mgt-suffixes st)))
               (new-roots (remove-assoc-equal word (mgt-roots st)))
               (st1 (mgt-set-tok2id new-tok2id st))
               (st2 (mgt-set-id2tok new-id2tok st1))
               (st3 (mgt-set-anchors new-anchors st2))
               (st4 (mgt-set-prefixes new-prefixes st3))
               (st5 (mgt-set-suffixes new-suffixes st4))
               (st6 (mgt-set-roots new-roots st5)))
          st6)))))

(defthm mgt-remove-vocab-word-pad-noop
  (equal (mgt-remove-vocab-word *pad-word* st) st))

(defthm mgt-remove-vocab-word-unk-noop
  (equal (mgt-remove-vocab-word *unk-word* st) st))

(defthm mgt-remove-vocab-word-bos-noop
  (equal (mgt-remove-vocab-word *bos-word* st) st))

(defthm mgt-remove-vocab-word-eos-noop
  (equal (mgt-remove-vocab-word *eos-word* st) st))

(defthm mgt-remove-vocab-word-removes-from-tok2id
  (implies (and (not (equal word *pad-word*))
                (not (equal word *unk-word*))
                (not (equal word *bos-word*))
                (not (equal word *eos-word*)))
           (not (alist-contains word (mgt-tok2id (mgt-remove-vocab-word word st))))))

(defthm mgt-remove-vocab-word-removes-from-anchors
  (implies (and (not (equal word *pad-word*))
                (not (equal word *unk-word*))
                (not (equal word *bos-word*))
                (not (equal word *eos-word*)))
           (not (alist-contains word (mgt-anchors (mgt-remove-vocab-word word st))))))

(defthm mgt-remove-vocab-word-preserves-next-id
  (equal (mgt-next-id (mgt-remove-vocab-word word st))
         (mgt-next-id st)))

(defthm mgt-remove-vocab-word-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-remove-vocab-word word st))
         (mgt-bpe-pairs st)))

(defthm mgt-remove-preserves-other-tok2id
  (implies (and (not (equal w1 w2))
                (not (equal w2 *pad-word*))
                (not (equal w2 *unk-word*))
                (not (equal w2 *bos-word*))
                (not (equal w2 *eos-word*)))
           (equal (alist-contains w1 (mgt-tok2id (mgt-remove-vocab-word w2 st)))
                  (alist-contains w1 (mgt-tok2id st)))))

(defun mgt-encode-batch-aux (texts st acc)
  (declare (xargs :guard t
                  :measure (acl2-count texts)))
  (if (atom texts)
      (reverse acc)
    (let ((encoded (mgt-encode (car texts) st)))
      (mgt-encode-batch-aux (cdr texts) st (cons encoded acc)))))

(defthm true-listp-mgt-encode-batch-aux
  (implies (true-listp acc)
           (true-listp (mgt-encode-batch-aux texts st acc)))
  :rule-classes (:rewrite :type-prescription))

(defun mgt-encode-batch (texts st)
  (declare (xargs :guard t))
  (mgt-encode-batch-aux texts st nil))

(defthm true-listp-mgt-encode-batch
  (true-listp (mgt-encode-batch texts st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-encode-batch-nil
  (equal (mgt-encode-batch nil st) nil))

(defthm len-mgt-encode-batch
  (equal (len (mgt-encode-batch texts st))
         (len texts))
  :hints (("Goal" :in-theory (enable mgt-encode-batch))))

(defun mgt-decode-batch-aux (token-lists st acc)
  (declare (xargs :guard t
                  :measure (acl2-count token-lists)))
  (if (atom token-lists)
      (reverse acc)
    (let ((decoded (mgt-decode (car token-lists) st)))
      (mgt-decode-batch-aux (cdr token-lists) st (cons decoded acc)))))

(defthm true-listp-mgt-decode-batch-aux
  (implies (true-listp acc)
           (true-listp (mgt-decode-batch-aux token-lists st acc)))
  :rule-classes (:rewrite :type-prescription))

(defun mgt-decode-batch (token-lists st)
  (declare (xargs :guard t))
  (mgt-decode-batch-aux token-lists st nil))

(defthm true-listp-mgt-decode-batch
  (true-listp (mgt-decode-batch token-lists st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-decode-batch-nil
  (equal (mgt-decode-batch nil st) nil))

(defthm len-mgt-decode-batch
  (equal (len (mgt-decode-batch token-lists st))
         (len token-lists))
  :hints (("Goal" :in-theory (enable mgt-decode-batch))))

(defun mgt-merge-subwords (subword-lists)
  (declare (xargs :guard t
                  :measure (acl2-count subword-lists)))
  (if (atom subword-lists)
      nil
    (append (if (true-listp (car subword-lists))
                (car subword-lists)
              nil)
            (mgt-merge-subwords (cdr subword-lists)))))

(defthm true-listp-mgt-merge-subwords
  (true-listp (mgt-merge-subwords subs))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-merge-subwords-nil
  (equal (mgt-merge-subwords nil) nil))

(defthm mgt-merge-subwords-singleton
  (implies (true-listp s)
           (equal (mgt-merge-subwords (list s)) s)))

(defthm len-mgt-merge-subwords-le
  (<= (len (mgt-merge-subwords (list s1 s2)))
      (+ (len s1) (len s2)))
  :rule-classes :linear)

(defun mgt-tokenize-with-anchors-aux (tokens st pos acc-anchors)
  (declare (xargs :guard (natp pos)
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      (list (reverse acc-anchors))
    (let* ((tid (car tokens))
           (tok-str (alist-get tid (mgt-id2tok st)))
           (tok-len (if tok-str (len tok-str) 0))
           (is-anchor (if tok-str (alist-contains tok-str (mgt-anchors st)) nil))
           (new-anchors (if is-anchor
                            (cons pos acc-anchors)
                          acc-anchors)))
      (mgt-tokenize-with-anchors-aux (cdr tokens) st (+ pos tok-len) new-anchors))))

(defun mgt-tokenize-with-anchors (text st)
  (declare (xargs :guard t))
  (let* ((tokens (mgt-encode text st))
         (anchor-info (mgt-tokenize-with-anchors-aux tokens st 0 nil)))
    (list tokens (car anchor-info))))

(defthm true-listp-mgt-tokenize-with-anchors-result
  (true-listp (mgt-tokenize-with-anchors text st))
  :rule-classes (:rewrite :type-prescription))

(defthm len-mgt-tokenize-with-anchors-result
  (equal (len (mgt-tokenize-with-anchors text st)) 2))

(defthm true-listp-first-of-tokenize-with-anchors
  (true-listp (car (mgt-tokenize-with-anchors text st))))

(defun mgt-detokenize (tokens st)
  (declare (xargs :guard t))
  (mgt-decode tokens st))

(defthm mgt-detokenize-is-decode
  (equal (mgt-detokenize tokens st)
         (mgt-decode tokens st)))

(defthm true-listp-mgt-detokenize
  (true-listp (mgt-detokenize tokens st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-detokenize-nil
  (equal (mgt-detokenize nil st) nil))

(defun mgt-unknown-replacement (context st)
  (declare (xargs :guard t)
           (ignore context st))
  *unk-id*)

(defthm mgt-unknown-replacement-is-unk
  (equal (mgt-unknown-replacement context st) *unk-id*))

(defthm natp-mgt-unknown-replacement
  (natp (mgt-unknown-replacement context st))
  :rule-classes (:rewrite :type-prescription))

(defun all-tokens-valid-p (tokens st)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      t
    (and (alist-contains (car tokens) (mgt-id2tok st))
         (all-tokens-valid-p (cdr tokens) st))))

(defthm all-tokens-valid-p-is-validate
  (equal (all-tokens-valid-p tokens st)
         (mgt-validate-tokens tokens st)))

(defthm all-tokens-valid-p-nil
  (equal (all-tokens-valid-p nil st) t))

(defthm all-tokens-valid-p-cons
  (equal (all-tokens-valid-p (cons tok rest) st)
         (and (alist-contains tok (mgt-id2tok st))
              (all-tokens-valid-p rest st))))

(defthm all-tokens-valid-p-append
  (equal (all-tokens-valid-p (append a b) st)
         (and (all-tokens-valid-p a st)
              (all-tokens-valid-p b st)))
  :hints (("Goal" :induct (all-tokens-valid-p a st))))

(defun mgt-add-prefix (prefix-word st)
  (declare (xargs :guard t))
  (let* ((result (mgt-add-token prefix-word st))
         (tid (car result))
         (st2 (cadr result))
         (new-prefixes (alist-put prefix-word tid (mgt-prefixes st2))))
    (mgt-set-prefixes new-prefixes st2)))

(defthm mgt-add-prefix-has-prefix
  (alist-contains prefix-word (mgt-prefixes (mgt-add-prefix prefix-word st))))

(defthm mgt-add-prefix-has-token
  (alist-contains prefix-word (mgt-tok2id (mgt-add-prefix prefix-word st))))

(defthm mgt-add-prefix-preserves-suffixes
  (equal (mgt-suffixes (mgt-add-prefix prefix-word st))
         (mgt-suffixes st)))

(defthm mgt-add-prefix-preserves-roots
  (equal (mgt-roots (mgt-add-prefix prefix-word st))
         (mgt-roots st)))

(defthm mgt-add-prefix-preserves-anchors
  (equal (mgt-anchors (mgt-add-prefix prefix-word st))
         (mgt-anchors st)))

(defthm mgt-add-prefix-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-add-prefix prefix-word st))
         (mgt-bpe-pairs st)))

(defun mgt-add-suffix (suffix-word st)
  (declare (xargs :guard t))
  (let* ((result (mgt-add-token suffix-word st))
         (tid (car result))
         (st2 (cadr result))
         (new-suffixes (alist-put suffix-word tid (mgt-suffixes st2))))
    (mgt-set-suffixes new-suffixes st2)))

(defthm mgt-add-suffix-has-suffix
  (alist-contains suffix-word (mgt-suffixes (mgt-add-suffix suffix-word st))))

(defthm mgt-add-suffix-has-token
  (alist-contains suffix-word (mgt-tok2id (mgt-add-suffix suffix-word st))))

(defthm mgt-add-suffix-preserves-prefixes
  (equal (mgt-prefixes (mgt-add-suffix suffix-word st))
         (mgt-prefixes st)))

(defthm mgt-add-suffix-preserves-roots
  (equal (mgt-roots (mgt-add-suffix suffix-word st))
         (mgt-roots st)))

(defthm mgt-add-suffix-preserves-anchors
  (equal (mgt-anchors (mgt-add-suffix suffix-word st))
         (mgt-anchors st)))

(defthm mgt-add-suffix-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-add-suffix suffix-word st))
         (mgt-bpe-pairs st)))

(defun mgt-add-prefixes (prefix-list st)
  (declare (xargs :guard t
                  :measure (acl2-count prefix-list)))
  (if (atom prefix-list)
      st
    (mgt-add-prefixes (cdr prefix-list)
                      (mgt-add-prefix (car prefix-list) st))))

(defthm mgt-add-prefixes-nil
  (equal (mgt-add-prefixes nil st) st))

(defthm mgt-add-prefixes-preserves-suffixes
  (equal (mgt-suffixes (mgt-add-prefixes plist st))
         (mgt-suffixes st))
  :hints (("Goal" :induct (mgt-add-prefixes plist st))))

(defthm mgt-add-prefixes-preserves-roots
  (equal (mgt-roots (mgt-add-prefixes plist st))
         (mgt-roots st))
  :hints (("Goal" :induct (mgt-add-prefixes plist st))))

(defthm mgt-add-prefixes-preserves-anchors
  (equal (mgt-anchors (mgt-add-prefixes plist st))
         (mgt-anchors st))
  :hints (("Goal" :induct (mgt-add-prefixes plist st))))

(defthm mgt-add-prefixes-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-add-prefixes plist st))
         (mgt-bpe-pairs st))
  :hints (("Goal" :induct (mgt-add-prefixes plist st))))

(defun mgt-add-suffixes (suffix-list st)
  (declare (xargs :guard t
                  :measure (acl2-count suffix-list)))
  (if (atom suffix-list)
      st
    (mgt-add-suffixes (cdr suffix-list)
                      (mgt-add-suffix (car suffix-list) st))))

(defthm mgt-add-suffixes-nil
  (equal (mgt-add-suffixes nil st) st))

(defthm mgt-add-suffixes-preserves-prefixes
  (equal (mgt-prefixes (mgt-add-suffixes slist st))
         (mgt-prefixes st))
  :hints (("Goal" :induct (mgt-add-suffixes slist st))))

(defthm mgt-add-suffixes-preserves-roots
  (equal (mgt-roots (mgt-add-suffixes slist st))
         (mgt-roots st))
  :hints (("Goal" :induct (mgt-add-suffixes slist st))))

(defthm mgt-add-suffixes-preserves-anchors
  (equal (mgt-anchors (mgt-add-suffixes slist st))
         (mgt-anchors st))
  :hints (("Goal" :induct (mgt-add-suffixes slist st))))

(defthm mgt-add-suffixes-preserves-bpe-pairs
  (equal (mgt-bpe-pairs (mgt-add-suffixes slist st))
         (mgt-bpe-pairs st))
  :hints (("Goal" :induct (mgt-add-suffixes slist st))))

(defun mgt-encode-to-rationals (text st)
  (declare (xargs :guard t
                  :measure (acl2-count text)))
  (let ((tokens (mgt-encode text st)))
    (mgt-tokens-to-rationals tokens)))

(defun mgt-tokens-to-rationals (tokens)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      nil
    (cons (if (natp (car tokens))
              (car tokens)
            0)
          (mgt-tokens-to-rationals (cdr tokens)))))

(defthm true-listp-mgt-tokens-to-rationals
  (true-listp (mgt-tokens-to-rationals tokens))
  :rule-classes (:rewrite :type-prescription))

(defthm all-natp-mgt-tokens-to-rationals
  (all-natp (mgt-tokens-to-rationals tokens)))

(defthm len-mgt-tokens-to-rationals
  (equal (len (mgt-tokens-to-rationals tokens))
         (len tokens)))

(defthm mgt-tokens-to-rationals-nil
  (equal (mgt-tokens-to-rationals nil) nil))

(defthm mgt-tokens-to-rationals-cons
  (equal (mgt-tokens-to-rationals (cons a rest))
         (cons (if (natp a) a 0)
               (mgt-tokens-to-rationals rest))))

(defun mgt-rationals-to-tokens (rats)
  (declare (xargs :guard t
                  :measure (acl2-count rats)))
  (if (atom rats)
      nil
    (cons (if (natp (car rats))
              (car rats)
            *unk-id*)
          (mgt-rationals-to-tokens (cdr rats)))))

(defthm true-listp-mgt-rationals-to-tokens
  (true-listp (mgt-rationals-to-tokens rats))
  :rule-classes (:rewrite :type-prescription))

(defthm len-mgt-rationals-to-tokens
  (equal (len (mgt-rationals-to-tokens rats))
         (len rats)))

(defthm mgt-rationals-to-tokens-nil
  (equal (mgt-rationals-to-tokens nil) nil))

(defthm mgt-rationals-to-tokens-inverse-of-tokens-to-rationals
  (implies (all-natp tokens)
           (equal (mgt-rationals-to-tokens (mgt-tokens-to-rationals tokens))
                  tokens))
  :hints (("Goal" :induct (mgt-tokens-to-rationals tokens))))

(defun mgt-decode-from-rationals (rats st)
  (declare (xargs :guard t))
  (mgt-decode (mgt-rationals-to-tokens rats) st))

(defthm true-listp-mgt-decode-from-rationals
  (true-listp (mgt-decode-from-rationals rats st))
  :rule-classes (:rewrite :type-prescription))

(defthm mgt-decode-from-rationals-nil
  (equal (mgt-decode-from-rationals nil st) nil))

(defun max-token-id-in (tokens)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      0
    (max (if (natp (car tokens)) (car tokens) 0)
         (max-token-id-in (cdr tokens)))))

(defthm natp-max-token-id-in
  (natp (max-token-id-in tokens))
  :rule-classes (:rewrite :type-prescription))

(defthm max-token-id-in-nil
  (equal (max-token-id-in nil) 0))

(defthm max-token-id-in-ge-car
  (implies (and (consp tokens) (natp (car tokens)))
           (<= (car tokens) (max-token-id-in tokens)))
  :rule-classes :linear)

(defthm max-token-id-in-ge-cdr
  (implies (consp tokens)
           (<= (max-token-id-in (cdr tokens)) (max-token-id-in tokens)))
  :rule-classes :linear)

(defthm max-token-id-in-member
  (implies (and (member-equal tok tokens)
                (natp tok))
           (<= tok (max-token-id-in tokens)))
  :rule-classes :linear)

(defun count-tokens (tokens)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      0
    (+ 1 (count-tokens (cdr tokens)))))

(defthm natp-count-tokens
  (natp (count-tokens tokens))
  :rule-classes (:rewrite :type-prescription))

(defthm count-tokens-is-len
  (equal (count-tokens tokens) (len tokens)))

(defun count-unk-tokens (tokens)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      0
    (+ (if (equal (car tokens) *unk-id*) 1 0)
       (count-unk-tokens (cdr tokens)))))

(defthm natp-count-unk-tokens
  (natp (count-unk-tokens tokens))
  :rule-classes (:rewrite :type-prescription))

(defthm count-unk-tokens-nil
  (equal (count-unk-tokens nil) 0))

(defthm count-unk-tokens-le-len
  (<= (count-unk-tokens tokens) (len tokens))
  :rule-classes :linear)

(defthm count-unk-tokens-append
  (equal (count-unk-tokens (append a b))
         (+ (count-unk-tokens a) (count-unk-tokens b)))
  :hints (("Goal" :induct (count-unk-tokens a))))

(defthm count-unk-tokens-zero-no-unks
  (implies (and (equal (count-unk-tokens tokens) 0)
                (consp tokens))
           (not (equal (car tokens) *unk-id*))))

(defun token-frequency-aux (tok tokens count)
  (declare (xargs :guard (natp count)
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      count
    (token-frequency-aux tok (cdr tokens)
                         (if (equal tok (car tokens))
                             (+ 1 count)
                           count))))

(defthm natp-token-frequency-aux
  (implies (natp count)
           (natp (token-frequency-aux tok tokens count)))
  :rule-classes (:rewrite :type-prescription))

(defun token-frequency (tok tokens)
  (declare (xargs :guard t))
  (token-frequency-aux tok tokens 0))

(defthm natp-token-frequency
  (natp (token-frequency tok tokens))
  :rule-classes (:rewrite :type-prescription))

(defthm token-frequency-nil
  (equal (token-frequency tok nil) 0))

(defthm token-frequency-le-len
  (<= (token-frequency tok tokens) (len tokens))
  :rule-classes :linear
  :hints (("Goal" :in-theory (enable token-frequency))))

(defthm token-frequency-zero-not-member
  (implies (equal (token-frequency tok tokens) 0)
           (not (member-equal tok tokens)))
  :hints (("Goal" :in-theory (enable token-frequency))))

(defun mgt-is-special-token-id (tid)
  (declare (xargs :guard t))
  (or (equal tid *pad-id*)
      (equal tid *unk-id*)
      (equal tid *bos-id*)
      (equal tid *eos-id*)))

(defthm booleanp-mgt-is-special-token-id
  (or (equal (mgt-is-special-token-id tid) t)
      (equal (mgt-is-special-token-id tid) nil))
  :rule-classes :type-prescription)

(defthm pad-is-special
  (equal (mgt-is-special-token-id *pad-id*) t))

(defthm unk-is-special
  (equal (mgt-is-special-token-id *unk-id*) t))

(defthm bos-is-special
  (equal (mgt-is-special-token-id *bos-id*) t))

(defthm eos-is-special
  (equal (mgt-is-special-token-id *eos-id*) t))

(defthm four-is-not-special
  (equal (mgt-is-special-token-id 4) nil))

(defun count-special-tokens (tokens)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      0
    (+ (if (mgt-is-special-token-id (car tokens)) 1 0)
       (count-special-tokens (cdr tokens)))))

(defthm natp-count-special-tokens
  (natp (count-special-tokens tokens))
  :rule-classes (:rewrite :type-prescription))

(defthm count-special-tokens-nil
  (equal (count-special-tokens nil) 0))

(defthm count-special-tokens-le-len
  (<= (count-special-tokens tokens) (len tokens))
  :rule-classes :linear)

(defthm count-special-tokens-append
  (equal (count-special-tokens (append a b))
         (+ (count-special-tokens a) (count-special-tokens b)))
  :hints (("Goal" :induct (count-special-tokens a))))

(defun filter-special-tokens (tokens)
  (declare (xargs :guard t
                  :measure (acl2-count tokens)))
  (if (atom tokens)
      nil
    (if (mgt-is-special-token-id (car tokens))
        (filter-special-tokens (cdr tokens))
      (cons (car tokens) (filter-special-tokens (cdr tokens))))))

(defthm true-listp-filter-special-tokens
  (true-listp (filter-special-tokens tokens))
  :rule-classes (:rewrite :type-prescription))

(defthm filter-special-tokens-nil
  (equal (filter-special-tokens nil) nil))

(defthm len-filter-special-tokens-le
  (<= (len (filter-special-tokens tokens)) (len tokens))
  :rule-classes :linear)

(defthm filter-special-tokens-no-specials
  (implies (consp (filter-special-tokens tokens))
           (not (mgt-is-special-token-id (car (filter-special-tokens tokens))))))

(defthm filter-special-tokens-append
  (equal (filter-special-tokens (append a b))
         (append (filter-special-tokens a) (filter-special-tokens b)))
  :hints (("Goal" :induct (filter-special-tokens a))))

(defthm len-filter-plus-count-special
  (equal (+ (len (filter-special-tokens tokens))
            (count-special-tokens tokens))
         (len tokens))
  :hints (("Goal" :induct (filter-special-tokens tokens))))

(defun pad-tokens-to-len (tokens target-len pad-id)
  (declare (xargs :guard (and (natp target-len) (true-listp tokens))
                  :measure (nfix (- (nfix target-len) (len tokens)))))
  (if (or (not (natp target-len))
          (>= (len tokens) target-len))
      tokens
    (pad-tokens-to-len (append tokens (list pad-id))
                       target-len
                       pad-id)))

(defthm true-listp-pad-tokens-to-len
  (implies (true-listp tokens)
           (true-listp (pad-tokens-to-len tokens target-len pad-id)))
  :rule-classes (:rewrite :type-prescription))

(defthm len-pad-tokens-to-len-ge
  (implies (and (natp target-len) (true-listp tokens))
           (<= target-len (len (pad-tokens-to-len tokens target-len pad-id))))
  :rule-classes :linear)

(defthm pad-tokens-noop-when-long-enough
  (implies (and (natp target-len)
                (true-listp tokens)
                (>= (len tokens) target-len))
           (equal (pad-tokens-to-len tokens target-len pad-id)
                  tokens)))

(defun truncate-tokens-to-len (tokens target-len)
  (declare (xargs :guard (natp target-len)))
  (if (or (not (natp target-len))
          (<= (len tokens) target-len))
      tokens
    (take target-len tokens)))

(defthm len-truncate-tokens-to-len-le
  (implies (natp target-len)
           (<= (len (truncate-tokens-to-len tokens target-len)) target-len))
  :rule-classes :linear)

(defthm truncate-tokens-noop-when-short-enough
  (implies (and (natp target-len)
                (<= (len tokens) target-len))
           (equal (truncate-tokens-to-len tokens target-len)
                  tokens)))

(defun mgt-encode-with-bos-eos (text st)
  (declare (xargs :guard t))
  (let ((tokens (mgt-encode text st)))
    (cons *bos-id* (append tokens (list *eos-id*)))))

(defthm true-listp-mgt-encode-with-bos-eos
  (true-listp (mgt-encode-with-bos-eos text st))
  :rule-classes (:rewrite :type-prescription))

(defthm consp-mgt-encode-with-bos-eos
  (consp (mgt-encode-with-bos-eos text st)))

(defthm car-mgt-encode-with-bos-eos
  (equal (car (mgt-encode-with-bos-eos text st)) *bos-id*))

(defthm len-mgt-encode-with-bos-eos
  (equal (len (mgt-encode-with-bos-eos text st))
         (+ 2 (len (mgt-encode text st)))))

(defthm last-token-is-eos
  (implies (true-listp (mgt-encode text st))
           (equal (nth (+ 1 (len (mgt-encode text st)))
                       (mgt-encode-with-bos-eos text st))
                  *eos-id*)))

(defun mgt-decode-strip-bos-eos (tokens st)
  (declare (xargs :guard t))
  (mgt-decode tokens st))

(defthm mgt-decode-strip-bos-eos-is-decode
  (equal (mgt-decode-strip-bos-eos tokens st)
         (mgt-decode tokens st)))

(defun mgt-vocab-words (st)
  (declare (xargs :guard t))
  (alist-keys (mgt-tok2id st)))

(defthm true-listp-mgt-vocab-words
  (true-listp (mgt-vocab-words st))
  :rule-classes (:rewrite :type-prescription))

(defun mgt-vocab-ids (st)
  (declare (xargs :guard t))
  (alist-vals (mgt-tok2id st)))

(defthm true-listp-mgt-vocab-ids
  (true-listp (mgt-vocab-ids st))
  :rule-classes (:rewrite :type-prescription))

(defun mgt-is-anchor-p (word st)
  (declare (xargs :guard t))
  (alist-contains word (mgt-anchors st)))

(defthm booleanp-mgt-is-anchor-p
  (or (equal (mgt-is-anchor-p word st) t)
      (equal (mgt-is-anchor-p word st) nil))
  :rule-classes :type-prescription)

(defun mgt-is-prefix-p (word st)
  (declare (xargs :guard t))
  (alist-contains word (mgt-prefixes st)))

(defthm booleanp-mgt-is-prefix-p
  (or (equal (mgt-is-prefix-p word st) t)
      (equal (mgt-is-prefix-p word st) nil))
  :rule-classes :type-prescription)

(defun mgt-is-suffix-p (word st)
  (declare (xargs :guard t))
  (alist-contains word (mgt-suffixes st)))

(defthm booleanp-mgt-is-suffix-p
  (or (equal (mgt-is-suffix-p word st) t)
      (equal (mgt-is-suffix-p word st) nil))
  :rule-classes :type-prescription)

(defun mgt-is-root-p (word st)
  (declare (xargs :guard t))
  (alist-contains word (mgt-roots st)))

(defthm booleanp-mgt-is-root-p
  (or (equal (mgt-is-root-p word st) t)
      (equal (mgt-is-root-p word st) nil))
  :rule-classes :type-prescription)

(defthm mgt-add-prefix-makes-prefix
  (mgt-is-prefix-p pw (mgt-add-prefix pw st)))

(defthm mgt-add-suffix-makes-suffix
  (mgt-is-suffix-p sw (mgt-add-suffix sw st)))

(defthm mgt-add-vocab-word-anchor-makes-anchor
  (mgt-is-anchor-p w (mgt-add-vocab-word w t st)))

(defun mgt-get-token-id (word st)
  (declare (xargs :guard t))
  (alist-get word (mgt-tok2id st)))

(defun mgt-get-token-word (tid st)
  (declare (xargs :guard t))
  (alist-get tid (mgt-id2tok st)))

(defthm mgt-get-token-id-nil-when-not-present
  (implies (not (alist-contains word (mgt-tok2id st)))
           (equal (mgt-get-token-id word st) nil)))

(defthm mgt-get-token-word-nil-when-not-present
  (implies (not (alist-contains tid (mgt-id2tok st)))
           (equal (mgt-get-token-word tid st) nil)))

(defthm mgt-get-token-id-after-add
  (implies (mgt-statep st)
           (equal (mgt-get-token-id word (mgt-add-token-state word st))
                  (mgt-add-token-id word st))))

(defthm mgt-get-token-word-after-add-new
  (implies (and (mgt-statep st)
                (not (alist-contains word (mgt-tok2id st))))
           (equal (mgt-get-token-word (mgt-add-token-id word st)
                                      (mgt-add-token-state word st))
                  word)))

(defun mgt-has-bpe-pair-p (pair-key st)
  (declare (xargs :guard t))
  (alist-contains pair-key (mgt-bpe-pairs st)))

(defthm booleanp-mgt-has-bpe-pair-p
  (or (equal (mgt-has-bpe-pair-p pk st) t)
      (equal (mgt-has-bpe-pair-p pk st) nil))
  :rule-classes :type-prescription)

(defun mgt-add-bpe-pair (pair-key token-id priority st)
  (declare (xargs :guard t))
  (let ((new-bpe (alist-put pair-key (cons token-id priority)
                            (mgt-bpe-pairs st))))
    (mgt-set-bpe-pairs new-bpe st)))

(defthm mgt-add-bpe-pair-present
  (mgt-has-bpe-pair-p pk (mgt-add-bpe-pair pk tid pri st)))

(defthm mgt-add-bpe-pair-preserves-tok2id
  (equal (mgt-tok2id (mgt-add-bpe-pair pk tid pri st))
         (mgt-tok2id st)))

(defthm mgt-add-bpe-pair-preserves-id2tok
  (equal (mgt-id2tok (mgt-add-bpe-pair pk tid pri st))
         (mgt-id2tok st)))

(defthm mgt-add-bpe-pair-preserves-prefixes
  (equal (mgt-prefixes (mgt-add-bpe-pair pk tid pri st))
         (mgt-prefixes st)))

(defthm mgt-add-bpe-pair-preserves-suffixes
  (equal (mgt-suffixes (mgt-add-bpe-pair pk tid pri st))
         (mgt-suffixes st)))

(defthm mgt-add-bpe-pair-preserves-roots
  (equal (mgt-roots (mgt-add-bpe-pair pk tid pri st))
         (mgt-roots st)))

(defthm mgt-add-bpe-pair-preserves-anchors
  (equal (mgt-anchors (mgt-add-bpe-pair pk tid pri st))
         (mgt-anchors st)))

(defthm mgt-add-bpe-pair-preserves-next-id
  (equal (mgt-next-id (mgt-add-bpe-pair pk tid pri st))
         (mgt-next-id st)))

(defun mgt-state-invariant (st)
  (declare (xargs :guard t))
  (and (mgt-statep st)
       (alist-contains *pad-word* (mgt-tok2id st))
       (alist-contains *unk-word* (mgt-tok2id st))
       (alist-contains *bos-word* (mgt-tok2id st))
       (alist-contains *eos-word* (mgt-tok2id st))))

(defthm mgt-state-invariant-implies-statep
  (implies (mgt-state-invariant st)
           (mgt-statep st))
  :rule-classes (:rewrite :forward-chaining))

(defthm mgt-state-invariant-has-pad
  (implies (mgt-state-invariant st)
           (alist-contains *pad-word* (mgt-tok2id st))))

(defthm mgt-state-invariant-has-unk
  (implies (mgt-state-invariant st)
           (alist-contains *unk-word* (mgt-tok2id st))))

(defthm mgt-state-invariant-has-bos
  (implies (mgt-state-invariant st)
           (alist-contains *bos-word* (mgt-tok2id st))))

(defthm mgt-state-invariant-has-eos
  (implies (mgt-state-invariant st)
           (alist-contains *eos-word* (mgt-tok2id st))))

(defthm mgt-add-token-preserves-invariant-tok2id-pad
  (implies (alist-contains *pad-word* (mgt-tok2id st))
           (alist-contains *pad-word* (mgt-tok2id (mgt-add-token-state word st)))))

(defthm mgt-add-token-preserves-invariant-tok2id-unk
  (implies (alist-contains *unk-word* (mgt-tok2id st))
           (alist-contains *unk-word* (mgt-tok2id (mgt-add-token-state word st)))))

(defthm mgt-add-token-preserves-invariant-tok2id-bos
  (implies (alist-contains *bos-word* (mgt-tok2id st))
           (alist-contains *bos-word* (mgt-tok2id (mgt-add-token-state word st)))))

(defthm mgt-add-token-preserves-invariant-tok2id-eos
  (implies (alist-contains *eos-word* (mgt-tok2id st))
           (alist-contains *eos-word* (mgt-tok2id (mgt-add-token-state word st)))))

(defthm mgt-remove-preserves-pad
  (alist-contains *pad-word*
                  (mgt-tok2id (mgt-remove-vocab-word word st))))

(defthm mgt-remove-preserves-unk
  (alist-contains *unk-word*
                  (mgt-tok2id (mgt-remove-vocab-word word st))))

(defthm mgt-remove-preserves-bos
  (alist-contains *bos-word*
                  (mgt-tok2id (mgt-remove-vocab-word word st))))

(defthm mgt-remove-preserves-eos
  (alist-contains *eos-word*
                  (mgt-tok2id (mgt-remove-vocab-word word st))))

(defun mgt-next-id-monotonic-witness (words st)
  (declare (xargs :guard t
                  :measure (acl2-count words)))
  (if (atom words)
      (nfix (mgt-next-id st))
    (mgt-next-id-monotonic-witness
     (cdr words)
     (mgt-add-token-state (car words) st))))

(defthm mgt-next-id-monotonic-witness-ge
  (implies (natp (mgt-next-id st))
           (<= (mgt-next-id st)
               (mgt-next-id-monotonic-witness words st)))
  :rule-classes :linear
  :hints (("Goal" :induct (mgt-next-id-monotonic-witness words st))))

(defun encode-deterministic-p (text st1 st2)
  (declare (xargs :guard t))
  (implies (and (equal (mgt-tok2id st1) (mgt-tok2id st2))
                (equal (mgt-id2tok st1) (mgt-id2tok st2))
                (equal (mgt-prefixes st1) (mgt-prefixes st2))
                (equal (mgt-suffixes st1) (mgt-suffixes st2))
                (equal (mgt-roots st1) (mgt-roots st2))
                (equal (mgt-bpe-pairs st1) (mgt-bpe-pairs st2)))
           (equal (mgt-encode text st1)
                  (mgt-encode text st2))))

(defthm encode-deterministic
  (encode-deterministic-p text st1 st2)
  :hints (("Goal" :in-theory (enable encode-deterministic-p))))

(defun decode-deterministic-p (tokens st1 st2)
  (declare (xargs :guard t))
  (implies (equal (mgt-id2tok st1) (mgt-id2tok st2))
           (equal (mgt-decode tokens st1)
                  (mgt-decode tokens st2))))

(defthm decode-deterministic
  (decode-deterministic-p tokens st1 st2)
  :hints (("Goal" :in-theory (enable decode-deterministic-p)
           :induct (mgt-decode tokens st1))))

(defthm mgt-encode-empty-text
  (equal (mgt-encode nil st) nil))

(defthm mgt-decode-empty-tokens
  (equal (mgt-decode nil st) nil))

(defthm mgt-subword-split-empty
  (equal (mgt-subword-split nil st) nil))

(defthm mgt-morph-decompose-empty
  (equal (mgt-morph-decompose nil st) nil))

(defthm mgt-longest-match-empty
  (equal (mgt-longest-match nil st) 0))

(defthm mgt-coverage-empty
  (equal (mgt-coverage nil st) 0))

(defthm mgt-validate-tokens-empty
  (equal (mgt-validate-tokens nil st) t))

(defthm mgt-encode-batch-empty
  (equal (mgt-encode-batch nil st) nil))

(defthm mgt-decode-batch-empty
  (equal (mgt-decode-batch nil st) nil))

(defthm mgt-merge-subwords-empty
  (equal (mgt-merge-subwords nil) nil))

(defthm filter-special-tokens-empty
  (equal (filter-special-tokens nil) nil))

(defthm count-special-tokens-empty
  (equal (count-special-tokens nil) 0))

(defthm count-unk-tokens-empty
  (equal (count-unk-tokens nil) 0))

(defthm mgt-encode-preserves-structure
  (true-listp (mgt-encode text st)))

(defthm mgt-decode-preserves-structure
  (true-listp (mgt-decode tokens st)))

(defthm mgt-subword-split-preserves-structure
  (true-listp (mgt-subword-split word st)))

(defthm mgt-encode-word-preserves-structure
  (true-listp (mgt-encode-word word st)))

(defthm mgt-morph-decompose-preserves-structure
  (true-listp (mgt-morph-decompose word st)))

(defthm mgt-tokens-to-rationals-preserves-structure
  (true-listp (mgt-tokens-to-rationals tokens)))

(defthm mgt-rationals-to-tokens-preserves-structure
  (true-listp (mgt-rationals-to-tokens rats)))

(defthm len-mgt-encode-batch-equals-input
  (equal (len (mgt-encode-batch texts st))
         (len texts)))

(defthm len-mgt-decode-batch-equals-input
  (equal (len (mgt-decode-batch token-lists st))
         (len token-lists)))

(defthm mgt-encode-with-bos-eos-len
  (equal (len (mgt-encode-with-bos-eos text st))
         (+ 2 (len (mgt-encode text st)))))

(defthm mgt-coverage-bounded
  (and (<= 0 (mgt-coverage text st))
       (<= (mgt-coverage text st) 1))
  :rule-classes nil)

(defthm mgt-vocab-size-non-negative
  (<= 0 (mgt-vocab-size st))
  :rule-classes :linear)

(defthm mgt-unknown-replacement-constant
  (equal (mgt-unknown-replacement ctx st) 1))

(defthm mgt-encode-word-nonempty-for-nonempty-input
  (implies (consp word)
           (consp (mgt-encode-word word st))))

(defthm mgt-subword-split-nonempty-for-nonempty-input
  (implies (consp word)
           (consp (mgt-subword-split word st))))