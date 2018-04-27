import xgboost as xgb
import random

class XGBLearner:
    def __init__(self, dtrain, evals, maximize, eta, objective,
                 metric, multi=False, seed=None, verbose_eval=False):
        self.maximize = maximize
        self.eta = eta
        self.objective = objective
        self.metric = metric
        self.dtrain = dtrain
        self.evals = evals
        self.verbose_eval = verbose_eval
        if seed is not None: self.seed = seed

        self.params = {}
        # model init
        self.params['objective'] = self.objective
        self.params['eta'] = self.eta
        self.params['eval_metric'] = self.metric
        self.params['seed'] = 20
        if multi: self.params['num_class'] = len(np.unique(self.dtrain.get_label()))

        # model complexity
        self.params['max_depth'] = 5
        self.params['min_child_weight'] = 10
        self.params['gamma'] = 1

        # noise and variance
        self.params['subsample'] = 1
        self.params['colsample_bytree'] = 1
        self.params['colsample_bylevel'] = 1

    def initialize(self):
        # run to find best iteration
        model = xgb.train(params=self.params,
                          dtrain=self.dtrain,
                          evals=self.evals,
                          maximize=self.maximize,
                          num_boost_round=int(10 // self.eta),
                          early_stopping_rounds=int(1 // (10 * self.eta)),
                          verbose_eval=self.verbose_eval)
        self.num_rounds = model.best_iteration
        if self.num_rounds > 0:
            print('Initialization Successful')
        else:
            print('Initialization not Successful, Retry by changing initial params')

    def max_depth_generator(limits):
        """upper and lower limits"""
        return NotImplementedError

    def random_params(self, seed=None):
        """
        Generates random parameters for random search
        """
        if seed is not None: random.setstate(seed)

        max_depth = random.randint(3, 20)  # max depth
        min_c_w = random.randint(1, 100)  # min child weight
        gamma = random.randint(1, 10)  # gamma
        subsample = random.uniform(0, 1)  # subsample
        colsample_bytree = random.uniform(0, 1)  # colsample by tree
        colsample_bylevel = random.uniform(0, 1)  # colsample by level

        # model complexity
        self.params['max_depth'] = max_depth
        self.params['min_child_weight'] = min_c_w
        self.params['gamma'] = gamma

        # noise and variance
        self.params['subsample'] = subsample
        self.params['colsample_bytree'] = colsample_bytree
        self.params['colsample_bylevel'] = colsample_bylevel

    def random_search(self, n_search, verbose=False):
        """Do random search for n_search times"""
        if not hasattr(self, "best_params"):
            self.best_params = None
            self.best_iteration = None

        for i in range(n_search):
            # get new random params
            self.random_params()

            model = xgb.train(params=self.params,
                              dtrain=self.dtrain,
                              evals=self.evals,
                              maximize=self.maximize,
                              num_boost_round=max(1, int(self.num_rounds * 2)),
                              early_stopping_rounds=int(self.num_rounds // 2),
                              verbose_eval=False)

            params_score, params_iteration = model.best_score, model.best_iteration

            if i == 0: self.best_score = params_score

            if (self.maximize & (params_score > self.best_score)) | \
                    (~self.maximize & (params_score < self.best_score)):
                self.best_score = params_score
                self.best_iteration = params_iteration
                self.best_params = self.params.copy()
                if verbose: print(f"Found new best score {self.best_score}")
        if self.best_params is not None:
            print("----------------------------------------")
            print("Best params and best iteration are found")
            print("----------------------------------------")
        else:
            print("----------------------------------------")
            print("Run more searches or retry with different initial params")
            print("----------------------------------------")

    def fit_best_model(self, dtrain):
        """fits best model based on random search"""
        self.best_model = xgb.train(params=self.best_params,
                                    dtrain=dtrain,
                                    num_boost_round=self.best_iteration,
                                    verbose_eval=False)

    def predict(self, dtest):
        return self.best_model.predict(dtest)
