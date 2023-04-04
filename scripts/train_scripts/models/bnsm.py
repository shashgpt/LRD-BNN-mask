from scripts.all_imports import *


# Prior
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = tfp.distributions.Normal(loc=0., scale=1.)

    @property
    def sigma(self):
        return tf.math.log1p(tf.exp(self.rho))
    
    def sample(self):
        epsilon = tfp.distributions.Sample(self.normal, self.rho.get_shape()).sample()
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input): # Sum of log_posterior_loss for each weight
        return tf.math.reduce_sum(-math.log(math.sqrt(2 * math.pi))-tf.math.log(self.sigma)-((input - self.mu) ** 2)/(2 * self.sigma ** 2))
    
    def median(self, no_of_samples = 100):
        pdf =  tfp.distributions.Normal(self.mu, self.sigma)
        median = pdf.cdf(tf.constant([0.5]))
        return median


# Posterior
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = tfp.distributions.Normal(loc=0., scale=sigma1)
        self.gaussian2 = tfp.distributions.Normal(loc=0., scale=sigma2)
    
    def log_prob(self, input): # Sum of log_prior_loss for each weight
        prob1 = tf.math.exp(self.gaussian1.log_prob(input))
        prob2 = tf.math.exp(self.gaussian2.log_prob(input))
        return tf.math.reduce_sum((tf.math.log(self.pi * prob1 + (1-self.pi) * prob2)))


# Custom metrics
class predictive_uncertainty(tf.keras.metrics.Metric):
    def __init__(self, name='predictive_uncertainty', **kwargs):
        super(predictive_uncertainty, self).__init__(name=name, **kwargs)
        self.predictive_uncertainty = self.add_weight(name='predictive_uncertainty', initializer='zeros')

    def update_state(self, predictive_uncertainty, sample_weight=None):
        self.predictive_uncertainty.assign(predictive_uncertainty)

    def result(self):
        return self.predictive_uncertainty

    def reset_states(self):
        self.predictive_uncertainty.assign(0)

class elbo_loss(tf.keras.metrics.Metric):
    def __init__(self, name='elbo_loss', **kwargs):
        super(elbo_loss, self).__init__(name=name, **kwargs)
        self.elbo_loss = self.add_weight(name='elbo_loss', initializer='zeros')

    def update_state(self, elbo_loss, sample_weight=None):
        self.elbo_loss.assign(elbo_loss)

    def result(self):
        return self.elbo_loss

    def reset_states(self):
        self.elbo_loss.assign(0)

# Bayesian Dense Layer
class BayesianDense(tf.keras.layers.Layer):
    def __init__(self, input_dim, units) -> None:
        super().__init__()

        # Weight parameters
        w_init = tf.random_normal_initializer()
        self.weight_mu = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"), trainable=True,)
        self.weight_rho = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"), trainable=True,)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        b_init = tf.zeros_initializer()
        self.bias_mu = tf.Variable(initial_value=b_init(shape=(units), dtype="float32"), trainable=True,)
        self.bias_rho = tf.Variable(initial_value=b_init(shape=(units), dtype="float32"), trainable=True,)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(tf.constant([0.5]), tf.constant([math.exp(-0)]), tf.constant([math.exp(-0.6)]))
        self.bias_prior = ScaleMixtureGaussian(tf.constant([0.5]), tf.constant([math.exp(-0)]), tf.constant([math.exp(-0.6)]))
        self.log_prior = 0
        self.log_variational_posterior = 0

    def call(self, input, calculate_log_probs=False, training=None):
        weight = self.weight.sample()
        bias = self.bias.sample()
        if training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return tf.math.sigmoid(tf.matmul(input, weight) + bias)

# BNSM Model
class BNSM(tf.keras.Model):
    def __init__(self, args, word_vectors) -> None:
        super().__init__()
        self.args = args

        # metrics
        self.elbo_loss_metric = elbo_loss()
        self.predictive_uncertainty_metric = predictive_uncertainty()
        self.acc_tracker_per_epoch = tf.keras.metrics.BinaryAccuracy(name="accuracy")

        self.word_vectors = word_vectors
        self.word_embeddings = tf.keras.layers.Embedding(self.word_vectors.shape[0], self.word_vectors.shape[1], 
                                                        embeddings_initializer=tf.keras.initializers.Constant(self.word_vectors), 
                                                        trainable=self.args.fine_tune_word_embeddings, 
                                                        mask_zero=True, 
                                                        name="word_embeddings")
        self.lstm = tf.keras.layers.LSTM(self.args.sequence_layer_units, dropout=0.5, name="classifier", return_sequences = False)
        # self.dense = tf.keras.layers.Dense(1, activation='sigmoid', name='output')
        self.dense = BayesianDense(self.args.sequence_layer_units, 1)

    def call(self, input_data):
        word_embeddings = self.word_embeddings(input_data)
        out = self.lstm(word_embeddings)
        out = self.dense(out)
        return out

    def log_prior(self): # Loss term
        return self.BayesianDense.log_prior
    
    def log_variational_posterior(self): # Loss term
        return self.BayesianDense.log_variational_posterior
    
    def train_step(self, data):
        x,  y = data
        sentences = x[0]
        sentiment_labels = y[0]
        with tf.GradientTape() as tape: # Forward propagation and loss calculation
            
            model_outputs = tf.zeros([self.args.num_of_bayesian_samples, self.args.batch_size, 1])
            log_likelihoods = tf.zeros([self.args.num_of_bayesian_samples, self.args.batch_size, 1])
            log_priors = tf.zeros([self.args.num_of_bayesian_samples])
            log_variational_posteriors = tf.zeros([self.args.num_of_bayesian_samples])

            # Sampling
            for sample in range(self.args.num_of_bayesian_samples):

                # Calculating log_likelihood per sample
                model_output = tf.math.log(self.call(sentences))
                model_outputs[sample] = model_output
                log_likelihoods[sample] = tf.matmul(sentiment_labels, model_output)

                # Calculate Priors and Posteriors
                log_priors[sample] = self.log_prior()
                log_variational_posteriors[sample] = self.log_variational_posterior()

            # Averaging over samples
            log_likelihood = log_likelihoods.mean(0)
            model_output_mean = model_outputs.mean(0)
            model_uncertainty = sum(model_outputs)/len(model_outputs)
            log_prior = log_priors.mean()
            log_variational_posterior = log_variational_posteriors.mean()

            # Calculating the ELBO
            complexity_loss = (log_variational_posterior - log_prior)/self.args.batch_size
            negative_log_likelihood = -tf.math.reduce_sum(tf.math.reduce_sum(log_likelihood, dim=1))
            data_dependent_loss = negative_log_likelihood
            loss = complexity_loss + data_dependent_loss
        
        # Compute gradients and update parameters
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # calculate the metrics
        self.acc_tracker_per_epoch.update_state(y, model_output_mean)
        self.predictive_uncertainty_metric.update_state(model_uncertainty)
        self.elbo_loss_metric.update_state(loss)
        metrics = {"elbo_loss":self.elbo_loss_metric.result(),
                   "accuracy":self.acc_tracker_per_epoch.result(),
                   "predictive_uncertainty":self.predictive_uncertainty_metric.result()}
        return metrics

    @property
    def metrics(self):
        return [self.elbo_loss_metric, self.acc_tracker_per_epoch, self.predictive_uncertainty_metric]

    def model(self):
        input_data = tf.keras.Input(shape=(None,), dtype="int64")
        model = tf.keras.Model(inputs=[input_data], outputs=[self.call(input_data)])
        return model


        

