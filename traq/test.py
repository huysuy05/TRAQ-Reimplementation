from datasets import load_dataset

# Specify a config to satisfy the dataset builder.
ds = load_dataset("rlyapin/OpenTriviaQA", "general")
print(ds)
print(ds["train"][0])
