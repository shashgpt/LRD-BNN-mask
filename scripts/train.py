from scripts.all_imports import *
from scripts.train_scripts.dataset_division import Dataset_division
from scripts.train_scripts.word_vectors import Word_vectors
from scripts.train_scripts.models.bnsm import *



class Train(object):
    def __init__(self, args) -> None:
        self.args = args

    def vectorize(self, sentences, word_index):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        vocab = [key for key in word_index.keys()]
        vectorize_layer = tf.keras.layers.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab)
        return vectorize_layer(np.array(sentences)).numpy()

    def train_model(self, preprocessed_dataset: dict):

        # create word embeddings
        if os.path.exists(self.args.dataset_dir+self.args.word_embeddings+".npy") and os.path.exists(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle"):
            word_vectors = np.load(open(self.args.dataset_dir+self.args.word_embeddings+".npy", "rb"))
            word_index = pickle.load(open(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle", "rb"))
        else:
            word_vectors, word_index = Word_vectors(self.args).create_word_vectors(preprocessed_dataset)
            with open(self.args.dataset_dir+self.args.word_embeddings+".npy", "wb") as handle:
                np.save(handle, word_vectors)
            with open(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle", "wb") as handle:
                pickle.dump(word_index, handle)

        # create train, val and test_dataset
        train_dataset, val_dataset, test_dataset = Dataset_division(self.args).train_val_test_split(preprocessed_dataset, divide_into_rule_sections=True)

        # create model
        if self.args.model_name == "brnn":
            model = BNSM(self.args, word_vectors).model()    
            model.compile(optimizer="adam")
            model.summary(line_length = 150)

        # train model
        train_sentences = self.vectorize(train_dataset["sentence"], word_index)
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])
        val_sentences = self.vectorize(val_dataset["val_dataset"]["sentence"], word_index)
        val_sentiment_labels = np.array(val_dataset["val_dataset"]["sentiment_label"])
        train_dataset = (train_sentences, train_sentiment_labels)
        val_dataset = (val_sentences, val_sentiment_labels)
        model.fit(x=train_dataset[0], y=train_dataset[1],
                        epochs=self.args.train_epochs, 
                        batch_size=self.args.batch_size, 
                        validation_data=val_dataset,
                        shuffle=False)
    
        # # save the model
        # if not os.path.exists("assets/trained_models/"+self.args.model_name):
        #     os.makedirs("assets/trained_models/"+self.args.model_name)
        # model.save_weights("assets/trained_models/"+self.args.model_name+".h5")
    
    # def evaluate_model(self, preprocessed_dataset: dict):
    
    #     # create word embeddings
    #     if os.path.exists(self.args.dataset_dir+self.args.word_embeddings+".npy") and os.path.exists(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle"):
    #         word_vectors = np.load(open(self.args.dataset_dir+self.args.word_embeddings+".npy", "rb"))
    #         word_index = pickle.load(open(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle", "rb"))
    #     else:
    #         word_vectors, word_index = Word_vectors(self.args).create_word_vectors(preprocessed_dataset)
    #         with open(self.args.dataset_dir+self.args.word_embeddings+".npy", "wb") as handle:
    #             np.save(handle, word_vectors)
    #         with open(self.args.dataset_dir+self.args.word_embeddings+"_word_index.pickle", "wb") as handle:
    #             pickle.dump(word_index, handle)

    #     # create train, val and test dataset
    #     train_dataset, val_dataset, test_dataset = Dataset_division(self.args).train_val_test_split(preprocessed_dataset, divide_into_rule_sections=True)

    #     # evaluate model
    #     test_sentences = self.vectorize(test_dataset["sentence"], word_index)
    #     test_sentiment_labels = np.array(test_dataset["sentiment_label"])
    #     test_dataset = (test_sentences, test_sentiment_labels)

    #     # load the model
    #     brnn_model = BNSM(self.args, word_vectors).model()
    #     brnn_model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

    #     # run the evaluation
    #     evaluations = brnn_model.evaluate(x=test_dataset[0], y=test_dataset[1])


