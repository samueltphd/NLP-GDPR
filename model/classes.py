class User:
    def __init__(self, uid, aggregator, logger):
        self.uid = uid # user id
        self.rids_participated = [] # participated round ids -- list to maintain order
        self.aggregator = aggregator # aggregator that this user communicates with
        self.logger = logger # logger stores uid to the list of rids they participated in and the range of data they participated in
        # keep track of the data from the user, kept as a list of anything
        self.data = [] # data id has to be included in each element - i.e. [{id: 1, val: "This is a text sent by Nam"}]
        self.current_weights = None # assumption is that there are some sorts of current weights that you can update on your local machine


    def update_data(self, update_func):
        """
        Function to update the existing data. Allows for flexibility for different applications to update user's data
        """
        self.data = update_func(self.data)


    def update_current_weights(self, update_func):
        """
        Function for the aggregator to send back the weights updates to the user to update the current weights.
        How it will do this is it will use the update function inputted in to update the current weights
        // TODO: Look for whether this is actually how it is done
        """
        self.current_weights = update_func(self.current_weights)
        return value


    def update_data_participated(self, rid):
        """
        function to request the aggregator to update the user's participation in the rid round of training
        """
        assert rid in self.rids_participated
        self.aggregator.update_user_participation_in_round(self.uid, rid)
    
    
    def remove_self_from_training(self, rid):
        """
        function to request the aggregator to remove them from a round of training
        input: round id that they are participating in
        """
        assert rid in self.rids_participated
        self.aggregator.remove_user_from_round(self.uid, rid)
        self.rids_participated.remove(rid)


    def train(self, round):
        """
        function to ask the user to train based on their local data
        input: - round: A Round, with a training function that the user will do,
                        and a data function that gets any subset of data from the user
                        to participate in the data
        output: whatever is required of the user from the aggregator, usually weights, to
                update the global model
        """
        # get the training function and data function
        train_f, data_f, rid = round.get_training_function(), round.get_data_function()
        # getting the data to be trained
        training_data = data_f(self.data)
        # getting the weights from the training function
        output = train_f(training_data)
        # ask the logger to log the round id 
        self.logger.log_round_participated(uid, rid, output)
        # return the data so the aggregator can get it
        return output




class Round:
    def __init__(self, rid, training_function, data_function):
        # instantiating a round with round ids
        self.rid = rid
        # training function:
        # - input: the data that the data function outputs, AND that takes into account existing weights from the user
        # - output: the weights that are going to be sent back to the aggregator
        self.training_function = training_function
        # data function:
        # - input: a list (always) that contains all the data that is stored on user's application
        # - output: a subset of that list that will be sent to the training function
        self.data_function = data_function


    def get_training_function(self):
        return self.training_function

    def get_data_function(self):
        return self.get_data_function



class Aggregator:
    def __init__(self, logger):
        self.logger = logger
        pass

    def remove_user_from_round(self, uid: int, rid: int):
        # TODO
        pass

    def update_user_participation_in_round(uid: int, rid: int):
        # TODO
        pass




class Log:
    def __init__(self):
        self.rounds = {} # rid to round
        self.uid_to_rids = {} # variable to map from uid to id of rounds of training (rids)
        self.rid_to_uids = {} # variable to map from rid to uids participated in that round
        self.uid_to_weights = {} # variable to keep track of the data indices that participated in the training
        # this data structure is to keep track of the weights contributed
        self.uid_rid_to_weights = {} # key: tuple, (uid, rid), value: the weights contributed


    def log_round_participated(self, uid, rid, weights): 
        """
        function to update the log to note that uid participated in rid
        input: uid (int), rid (int)
        output: None
        TODO: Having to log the weights that they participated might reveal information about them
        """
        if uid not in self.uid_to_rids: self.uid_to_rids[uid] = [rid]
        else: self.uid_to_rids[uid].append(rid)
        if rid not in self.rid_to_uids: self.rid_to_uids[rid] = [uid]
        else: self.rid_to_uids[rid].append(uid)
        # adding weights that uid and rid that contributed
        self.uid_rid_to_weights[(uid, rid)] = weights


    def delete_round_participated(self, uid, rid):
        """
        function that delete a user's participation in the round
        """
        self.uid_to_rids[uid].remove(rid)
        self.rid_to_uids[rid].remove(uid)
        del self.uid_rid_to_weights[(uid, rid)]
        

    def log_round(self, rid, t_round):
        """
        function to keep track of the rounds
        input: rid (int), round (Round)
        output: None
        """
        if rid in self.rounds: raise Exception("round is already logged.")
        self.rounds[rid] = t_round


    def get_round(self, rid):
        if not rid in self.rounds: raise Exception("round is not in the dict. uh oh.")
        return self.rounds[rid]

    
