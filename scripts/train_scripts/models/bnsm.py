from scripts.all_imports import *



DEVICE = torch.device(0)
# Prior
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)
    
    def log_prob(self, input): # Sum of log_prior_loss for each weight
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
    

# Posterior
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input): # Sum of log_posterior_loss for each weight
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
    
    def median(self, no_of_samples = 100):
        pdf =  torch.distributions.Normal(self.mu, self.sigma)
        median = pdf.icdf(torch.Tensor([0.5])).to(DEVICE)
        return median
    

# Bayesian Dense Layer
class BayesianDense(torch.nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()

        # Weight parameters
        self.weight_mu = torch.nn.Parameter(torch.Tensor(units, input_dim).uniform_(-0.2, 0.2)).to(DEVICE)
        self.weight_rho = torch.nn.Parameter(torch.Tensor(units, input_dim).uniform_(-5,-4)).to(DEVICE)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = torch.nn.Parameter(torch.Tensor(units).uniform_(-0.2, 0.2)).to(DEVICE)
        self.bias_rho = torch.nn.Parameter(torch.Tensor(units).uniform_(-5,-4)).to(DEVICE)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(torch.Tensor([0.5]).to(DEVICE), torch.Tensor([math.exp(-0)]).to(DEVICE), torch.Tensor([math.exp(-6)]).to(DEVICE))
        self.bias_prior = ScaleMixtureGaussian(torch.Tensor([0.5]).to(DEVICE), torch.Tensor([math.exp(-0)]).to(DEVICE), torch.Tensor([math.exp(-6)]).to(DEVICE))
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, calculate_log_probs=False, sample="random_sample"):
        if sample == "random_sample":
            weight = self.weight.sample()
            bias = self.bias.sample()
        elif sample == "median":
            weight = self.weight.median()
            bias = self.bias.median()
        elif sample == "mean":
            weight = self.weight.mu #mean of variational posterior on weights for particular layer
            bias = self.bias.mu #mean of variational posterior on bias for a particular layer 
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return torch.nn.functional.linear(input, weight, bias)

# Dataset
class Dataset_batching(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        sentence = self.x[index]
        label = self.y[index]
        return sentence, label

# BNN model
class BNN(torch.nn.Module):
    def __init__(self, args, word_vectors): # Architecture
        super().__init__()
        self.args = args
        word_vectors = torch.Tensor(word_vectors)

        self.embedding = torch.nn.Embedding(word_vectors.shape[0], word_vectors.shape[1], padding_idx=0)
        self.embedding.load_state_dict({'weight': word_vectors})
        self.embedding.weight.requires_grad = self.args.fine_tune_word_embeddings

        self.lstm = torch.nn.LSTM(input_size = word_vectors.shape[1], hidden_size = self.args.sequence_layer_units, batch_first=True)
        # self.dense = BayesianDense(self.args.sequence_layer_units, 2)
        self.dense = torch.nn.Linear(self.args.sequence_layer_units, 2)
    
    def forward(self, input_data): # Forward Propagation
        word_embeddings = self.embedding(input_data)
        out = self.lstm(word_embeddings)[0][:, -1, :]
        out = self.dense(out)
        return out

    def log_prior(self): # Loss term
        return self.dense.log_prior

    def log_variational_posterior(self): # Loss term
        return self.dense.log_variational_posterior

    def fit(self, train_dataset, val_dataset, model, optimizer, additional_validation_datasets=None):
        
        # Collects per-epoch loss and acc like keras fit()
        history = {} 
        history['train_loss'] = []
        history['train_accuracy'] = []
        history['train_uncertainty'] = []
        history['val_loss'] = []
        history['val_accuracy'] = []
        history['val_uncertainty'] = []

        # start the timer
        start_time_sec = time.time()

        # early stopping
        patience = 0
        best_val_acc = 0

        # Epoch
        for epoch in range(1, self.args.train_epochs+1):
            
            # batch the dataset
            train_dataset_batched = Dataset_batching(train_dataset[0], train_dataset[1], transform = None)
            train_loader = torch.utils.data.DataLoader(train_dataset_batched, batch_size = self.args.batch_size, shuffle=False)

            model.train()
            train_loss = 0.0
            num_train_correct  = 0
            num_train_examples = 0
            
            # Train iteration
            train_pbar = trange(len(train_loader), position=0, leave=True, desc='Iteration')
            for batch_idx in train_pbar:
                
                input_data, target = next(iter(train_loader))
                input_data, target = input_data.to(DEVICE), target.to(DEVICE)

                # # Forward pass
                # model_outputs = torch.zeros(self.args.num_of_bayesian_samples, input_data.shape[0], 2).to(DEVICE)
                # log_likelihoods = torch.zeros(self.args.num_of_bayesian_samples, input_data.shape[0], 2).to(DEVICE)
                # log_priors = torch.zeros(self.args.num_of_bayesian_samples).to(DEVICE)
                # log_variational_posteriors = torch.zeros(self.args.num_of_bayesian_samples).to(DEVICE)

                # # Sampling
                # for sample in range(self.args.num_of_bayesian_samples):
                #     model_outputs[sample] = torch.log_softmax(self(input_data), dim=1)
                #     ground_truth = torch.nn.functional.one_hot(target)
                #     log_likelihoods[sample] = torch.mul(ground_truth, model_outputs[sample])
                #     log_priors[sample] = self.log_prior()
                #     log_variational_posteriors[sample] = self.log_variational_posterior()
                
                # log_prior = log_priors.mean()
                # log_variational_posterior = log_variational_posteriors.mean()
                # log_likelihood = log_likelihoods.mean(0)
                # negative_log_likelihood = -torch.sum(torch.sum(log_likelihood, dim=1))
                # loss = (log_variational_posterior - log_prior)/len(train_loader) + negative_log_likelihood

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # train_loss_iter = loss.data.item()
                # pred = model_outputs.mean(0).max(1, keepdim=True)[1]
                # train_acc_iter = pred.eq(target.view_as(pred)).sum().item()/input_data.shape[0]

                # train_loss += train_loss_iter
                # num_train_correct += pred.eq(target.view_as(pred)).sum().item()
                # num_train_examples += input_data.shape[0]

                softmax = torch.nn.Softmax(dim=1)
                model_output = softmax(model(input_data))
                ground_truth = torch.nn.functional.one_hot(target)
                NLLLoss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))
                optimizer.zero_grad()
                NLLLoss.backward()
                optimizer.step()

                train_loss_iter = NLLLoss.data.item()
                train_acc_iter = ((torch.max(model_output, 1)[1] == target).sum().item())/input_data.shape[0]

                train_loss += NLLLoss.data.item() * input_data.size(0)
                num_train_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                num_train_examples += input_data.shape[0]

                train_pbar.set_description('train loss: %g, train acc: %g' % (train_loss_iter, train_acc_iter))
            
            train_acc = num_train_correct / num_train_examples
            train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)

            # Val Iteration
            model.eval()
            val_loss = 0.0
            num_val_correct  = 0
            num_val_examples = 0

            val_dataset_batched = Dataset_batching(val_dataset[0], val_dataset[1], transform = None)
            val_loader = torch.utils.data.DataLoader(val_dataset_batched, batch_size = self.args.batch_size, shuffle=False)

            with torch.no_grad():
                for batch_idx in range(len(val_loader)):
                    input_data, target = next(iter(val_loader))
                    input_data, target = input_data.to(DEVICE), target.to(DEVICE)

                    # # Forward pass
                    # model_outputs = torch.zeros(self.args.num_of_bayesian_samples, input_data.shape[0], 2).to(DEVICE)
                    # log_likelihoods = torch.zeros(self.args.num_of_bayesian_samples, input_data.shape[0], 2).to(DEVICE)
                    # log_priors = torch.zeros(self.args.num_of_bayesian_samples).to(DEVICE)
                    # log_variational_posteriors = torch.zeros(self.args.num_of_bayesian_samples).to(DEVICE)

                    # # Sampling
                    # for sample in range(self.args.num_of_bayesian_samples):
                    #     model_outputs[sample] = torch.log_softmax(self(input_data), dim=1)
                    #     ground_truth = torch.nn.functional.one_hot(target)
                    #     log_likelihoods[sample] = torch.mul(ground_truth, model_outputs[sample])
                    #     log_priors[sample] = self.log_prior()
                    #     log_variational_posteriors[sample] = self.log_variational_posterior()
                    
                    # log_prior = log_priors.mean()
                    # log_variational_posterior = log_variational_posteriors.mean()
                    # log_likelihood = log_likelihoods.mean(0)
                    # negative_log_likelihood = -torch.sum(torch.sum(log_likelihood, dim=1))
                    # loss = (log_variational_posterior - log_prior)/len(val_loader) + negative_log_likelihood

                    # val_loss += loss.data.item()
                    # pred = model_outputs.mean(0).max(1, keepdim=True)[1]
                    # num_val_correct += pred.eq(target.view_as(pred)).sum().item()
                    # num_val_examples += input_data.shape[0]

                    softmax = torch.nn.Softmax(dim=1)
                    model_output = softmax(model(input_data))
                    ground_truth = torch.nn.functional.one_hot(target)
                    NLLLoss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))

                    val_loss += NLLLoss.data.item() * input_data.size(0)
                    num_val_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                    num_val_examples += input_data.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
                
            print("\nEpoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f" % \
                    (epoch, self.args.train_epochs, history['train_loss'][epoch-1], history['train_accuracy'][epoch-1], history['val_loss'][epoch-1], history['val_accuracy'][epoch-1]))
                


