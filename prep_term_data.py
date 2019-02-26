import sys, os, codecs, joblib, argparse
import numpy as np
import utils

# Do we remove training and dev queries that don't have a pre-trained
# embedding? If not, we assign a random embedding.
REMOVE_OOV_TRAIN_QUERIES = True
REMOVE_OOV_DEV_QUERIES = True

doc = """ Prepare data to test a model on a given dataset,
and write in a pickle file. Added by johmy592"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=doc)

    msg = ("path to vocabulary file")
    parser.add_argument("path_vocab", help=msg)
    msg= ("path to file containing test queries")
    parser.add_argument("path_queries",help=msg)
    msg = ("path of text file containing embeddings of the queries and candidates")
    parser.add_argument("path_embeddings", help=msg)
    parser.add_argument("path_output", help="path of output (pickle file)")
    parser.add_argument("-s", "--seed", type=int, default=91500)
    args = parser.parse_args()


    print("Loading candidates...")
    path_candidates = args.path_vocab
    candidates = utils.load_candidates(path_candidates, normalize=True)
    print("Nb candidates: {}".format(len(candidates)))

    path_q_test = args.path_queries
    q_test, _ =  utils.load_queries(path_q_test, normalize=True)
    print("Nb test queries: {}".format(len(q_test)))


    print("Loading pre-trained word embeddings...")
    embed_vocab_list, word2vec = utils.get_embeddings(args.path_embeddings, np.float32)
    embed_vocab_set = set(embed_vocab_list)
    print("Nb embeddings: {}".format(len(embed_vocab_list)))


    print("Checking for candidates that don't have a pre-trained embedding...")
    oov_candidates = set(c for c in candidates if c not in embed_vocab_set)
    print("Nb candidates without a pre-trained embedding: {}".format(len(oov_candidates)))
    if len(oov_candidates):
        print("WARNING: {} candidates will be assigned a random embedding.".format(len(oov_candidates)))


    print("Checking for test queries that don't have a pre-trained embedding...")
    oov_query_ix_test = [i for i,q in enumerate(q_test) if q not in embed_vocab_set]
    print("Nb test queries without a pre-trained embedding: {}".format(len(oov_query_ix_test)))
    if len(oov_query_ix_test):
        m = ", ".join(q_test[i] for i in oov_query_ix_test)
        print("WARNING: these dev queries will be assigned a random embedding: {}".format(m))


    print("Making embedding array for candidates...")
    candidate_embeds = utils.make_embedding_matrix(word2vec, candidates, seed=args.seed)
    candidate_embeds = utils.normalize_numpy_matrix(candidate_embeds)
    print("Nb embeddings: {}".format(candidate_embeds.shape[0]))

    print("Making embedding array for test queries...")
    test_query_embeds = utils.make_embedding_matrix(word2vec, q_test, seed=args.seed)
    test_query_embeds = utils.normalize_numpy_matrix(test_query_embeds)
    print("Nb embeddings: {}".format(test_query_embeds.shape[0]))


    # Make array of (query IDs, hypernym ID) pairs
    print("Making array of (query ID, hypernym ID) pairs...")
    candidate_to_id = {w:i for i,w in enumerate(candidates)}

    test_q_cand_ids = [candidate_to_id[q] if q in candidate_to_id else None for q in q_test]

    data = {}
    data["candidates"] = candidates
    data["candidate_embeds"] = candidate_embeds
    data["test_queries"] = q_test
    data["test_query_embeds"] = test_query_embeds
    data["test_query_cand_ids"] = test_q_cand_ids


    print("\nData:")
    for k,v in data.items():
        print("- {} ({}.{})".format(k, type(v).__module__, type(v).__name__))
    joblib.dump(data, args.path_output)
    print("\nWrote data --> {}\n".format(args.path_output))
