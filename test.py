import pickle

f = open("lists.pkl", "rb")
lists = pickle.load(f)

test_list = lists.get("test_list")

