# different modes of GDPR compliance
NO_COMPLIANCE, VERY_WEAK, WEAK, NEUTRAL, STRONG, STRICT = 0, 1, 2, 3, 4, 5

class Round:
    def __init__(self, rid, training_function, data_function, aggregation_function, num_participating_devices):
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
        # num participating devices - for the aggregator to recruit
        self.num_participants = num_participants
        # aggregation function:
        # - input:
        #       + dict of weights from all the different devices. format:
        #               uid (int): weight (list-like)
        #       + a previous global set of weights (list-like)
        #
        # - output:
        #       + the result of aggregation, probably the checkpoint
        #       + a dict that maps from uid (int) to the weight changes function to be happened (that takes in
        #           the current set of weights from the user, and update it to be something)
        #           +-+-+ weight function: a function that takes in a previous set of weights and output a novel set of weights
        self.aggregation_function = aggregation_function


    def get_training_function(self):
        return self.training_function


    def get_data_function(self):
        return self.data_function


    def get_aggregation_function(self):
        return self.aggregation_function


    def get_round_id(self):
        return self.rid


    def get_num_participants(self):
        return self.num_participants