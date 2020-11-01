import random

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
        self.num_participating_devices = num_participating_devices
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
        return self.get_data_function


    def get_aggregation_function(self):
        return self.aggregation_function


    def get_round_id(self):
        return self.rid





"""
The design of this Logger is so that everything is logged only AFTER the changes have been made - i.e. only log that
user has been removed from a round when the training has already been updated
"""
class Log:
    def __init__(self):
        self.rounds = {} # rid to round
        self.uid_to_rids = {} # variable to map from uid to id of rounds of training (rids)
        self.rid_to_uids = {} # variable to map from rid to uids participated in that round
        # this data structure is to keep track of the weights contributed
        self.uid_rid_to_weights = {} # key: tuple, (uid, rid), value: the weights contributed
        self.uid_to_user = {} # this to add the users to the log
        self.rid_to_round = {} # this to add the rounds to the log
        self.rid_to_global_checkpoints = {} # this is to save the checkpoints after the training algorithm
        # the last variable for stuff
        self.next_rid = 0


    ############## FUNCTIONS TO USE TO CALCULATE THINGS FOR THE WEIGHT UPDATES ##################
    def weights_given_rid_excluding_uids(self, rid, excluding_uids=[]):
        # get the uids associated with this
        associated_uids = self.get_uids_from_rid(rid)
        return {uid: self.uid_rid_to_weights[(uid, rid)] for uid in associated_uids if uid not in excluding_uids}


    ##################### FUNCTIONS TO USE AFTER WEIGHTS UPDATES ARE MADE #######################
    # DONE
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


    # DONE
    def delete_round_participated(self, uid, rid):
        """
        function that delete a user's participation in the round
        """
        self.uid_to_rids[uid].remove(rid)
        self.rid_to_uids[rid].remove(uid)
        del self.uid_rid_to_weights[(uid, rid)]


    #################################### GETTERS AND SETTERS ####################################
    # DONE
    def add_round(self, rid, t_round):
        if rid in self.rid_to_round: raise Exception("rid already added")
        self.rid_to_round[rid] = t_round


    # DONE
    def add_user(self, uid, user):
        if uid in self.uid_to_user: raise Exception("uid already added.")
        self.uid_to_user[uid] = user


    # DONE
    def get_user(self, uid):
        if uid not in self.uid_to_user: return None
        return self.uid_to_user[uid]


    # DONE
    def get_round(self, rid):
        if rid not in self.rid_to_round: return None
        return self.rid_to_round[rid]


     # DONE
    def get_rids_from_uid(self, uid):
        if uid not in self.uid_to_rids: return None
        return self.uid_to_rids[uid]


    # DONE
    def get_uids_from_rid(self, rid):
        if rid not in self.rid_to_uids: return None
        return self.rid_to_uids[rid]


    # DONE
    def get_next_rid(self):
        return self.next_rid


    # DONE
    def set_global_checkpoint(self, rid, value, replace=False):
        if rid in self.rid_to_global_checkpoints and not replace:
            raise Exception("Trying to override global checkpoint without permission")
        self.rid_to_global_checkpoints[rid] = value
        return value


    # DONE
    def get_global_checkpoint(self, rid):
        if rid not in self.rid_to_global_checkpoints: return None
        return self.rid_to_global_checkpoints[rid]


    # DONE
    def get_weight_contributed_by_device(self, uid, rid):
        if (uid, rid) not in self.uid_rid_to_weights: return None
        return self.uid_rid_to_weights[(uid, rid)]


    def sample_users(num_selecting_users):
        # return a {uid (int): user (User)} dict
        random_ids = random.sample(self.uid_to_user.keys(), num_selecting_users)
        return {k: self.uid_to_user[k] for k in random_ids}
