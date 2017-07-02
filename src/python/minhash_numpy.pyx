cimport numpy as np
import numpy as np
cimport cython

cdef unsigned int PRIME = 3000000019
cdef unsigned int MAX_TWITTER_ID = 3000000000

ctypedef np.uint_t DTYPE_t
ctypedef np.uint32_t DTYPE32_t

cpdef process_fan(fan_stars, hash_vals, np.ndarray[DTYPE32_t, ndim=2] signatures):
    star_signatures_fans = signatures[fan_stars, :]
    star_signatures_fans = np.minimum(star_signatures_fans,
                                      hash_vals)  # Automaticaly replicates (broadcasts) the hash_val array column-wise
    signatures[fan_stars, :] = star_signatures_fans

cpdef np.ndarray hash_function(unsigned long fan_id, np.ndarray[np.uint_t, ndim=2] hash_a, np.ndarray[np.uint_t, ndim=2] hash_b):
    return np.mod(np.mod((hash_a*fan_id + hash_b), PRIME),MAX_TWITTER_ID)# Permute the row number (fan_id)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef calculate_minhashes(unsigned long[:, ::1] edges, np.ndarray[DTYPE32_t, ndim=2] signatures,
                          np.ndarray[DTYPE_t, ndim=2] hash_a, np.ndarray[DTYPE_t, ndim=2] hash_b,
                          int num_hashes):
    fan_stars = []
    cdef Py_ssize_t count
    cdef unsigned int edge_count = edges.shape[0]
    cdef unsigned int follower_id, influencer_index, follower_previous_id = edges[0, 0]
    cdef np.ndarray[np.uint_t, ndim=2] hashes = np.zeros((1, num_hashes), dtype=np.uint)

    for count in range(edge_count):
        print "edge count", edge_count
        # File format assumed: follower_id, influencer_id
        follower_id, influencer_index = edges[count, 0], edges[count, 1]

        # Build up list of Stars the follower follows
        if follower_id == follower_previous_id:
            fan_stars.append(influencer_index)
        # Process follower - this also happens on first line
        else:
            hashes = hash_function(follower_previous_id, hash_a, hash_b)

            # Obtain stars the follower follows
            process_fan(fan_stars, hashes, signatures)

            follower_previous_id = follower_id
            fan_stars = [influencer_index]
    # Process last fan
    hashes = hash_function(follower_previous_id, hash_a, hash_b)

    # Obtain stars the follower follows
    process_fan(fan_stars, hashes, signatures)