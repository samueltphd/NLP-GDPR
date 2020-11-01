import random

# different modes of GDPR compliance
NO_COMPLIANCE, VERY_WEAK, WEAK, NEUTRAL, STRONG, STRICT = 0, 1, 2, 3, 4, 5


class User:
    def __init__(self, uid, aggregator, logger, compliance_mode=VERY_WEAK):
        self.uid = uid # user id
        self.aggregator = aggregator # aggregator that this user communicates with
        self.logger = logger # logger stores uid to the list of rids they participated in and the range of data they participated in
        # keep track of the data from the user, kept as a list of anything
        self.data = [] # opt_our boolean has to be included in each element - i.e. [{opt_out: Tru, val: "This is a text sent by Nam"}]
        # structure for convenient rollbacks / history weight access
        self.rid_to_local_weight = {}
        # These two variables are to keep track of each data point's participation in each round
        # id -> dict
        self.data_id_to_data_point = {}
        # int -> lists
        self.rid_to_data_ids = {}
        self.data_id_to_rids = {}
        # this data structure is to keep track of local changes uncommitted to the global level
        # store all the round ids that are uncommitted
        self.uncommitted = []


    def add_data(self, value, type_check=True):
        if type_check:
            assert type(value) == dict
            assert "id" in value
            if "rids" not in value: value["rids"] = []
            if "opt_in" not in value: value["opt_in"]=True
            # assert that this data id has not been added before
            assert value["id"] not in self.data_id_to_data_point
        # add it to existing datastore
        self.data.append(value)
        # keep track of the pointer that points to this data poin
        self.data_id_to_data_point[value["id"]] = value

    
    def remove_data(self, data_id, deep=True):
        assert data_id in self.data_id_to_data_point
        data_point = self.data_id_to_data_point[data_id]
        # remove from the local data store
        self.data.remove(data_point)
        del self.data_id_to_data_point[data_id]
        # if deep, meaning the user want this local removal to be reflected on a global level,
        # we put this into a committed list then
        if deep:
            self.uncommitted.extend(data_point["rids"])
            self.uncommitted = list(set(self.uncommitted))


    def update_data(self, data_id, value, deep=True, type_check=True):
        assert data_id in self.data_id_to_data_point
        data_point = self.data_id_to_data_point[data_id]
        # assert that the value that we are replacing with also has the right types and values
        if type_check:
            assert type(value) == dict
            assert ("id" in value) and (value["id"] == data_point["id"])
            assert ("rids" in value) and (value["rids"] == data_point["rids"])
            assert ("opt_in" in value) and (value["opt_in"] == data_point["opt_in"])
        # now update the local data stores
        self.data.remove(data_point)
        self.data.append(value)
        self.data_id_to_data_point[data_id] = value
        # if deep, keep track of the changes
        if deep:
            self.uncommitted.extend(data_point["rids"])
            self.uncommitted = list(set(self.uncommitted))

    
    def get_opted_in_data(self):
        """ Get opted in data for training """
        return [e for e in self.data if e["opt_in"]]


    def update_rid_to_local_weight(self, rid, new_weight, replace=False):
        """
        Function for the aggregator to send back the weights updates to the user to update the current weights.
        How it will do this is it will use the update function inputted in to update the current weights
        // TODO: Look for whether this is actually how it is done
        """
        if rid in self.rid_to_local_weight and not replace:
            raise Exception("Trying to update user.rid_to_local_weights dict without the permission")
        self.rid_to_local_weight[rid] = new_weight


    def get_weight_from_rid(self, rid):
        if rid not in self.rid_to_local_weight: return None
        return self.rid_to_local_weight[rid]


    def update_data_participated(self, rid):
        """ function to request the aggregator to update the user's participation in the rid round of training """
        assert rid in self.logger.get_rids_from_uid(self.uid)
        self.aggregator.update_user_participation_in_round(self.uid, rid)


    def change_data_permission(self, data_id, value=False, deep=True):
        """
        opt data in or out of training.
        - value=False: opting the data associated with data_id out from future training
        - value=True: opting in
        """
        if data_id not in self.data_id_to_data_point:
            raise Exception("invalid data_id referred")
        data_point = self.data_id_to_data_point[data_id]
        data_point["opt_in"] = value
        if deep:
            self.uncommitted.extend(data_point["rids"])
            self.uncommitted = list(set(self.uncommitted))


    def request_aggregator_update(self):
        # from the uncommitted, get the relevant round ids
        to_update_round_ids = []
        self.aggregator.user_request_update(self.uid, self.uncommitted)
        # and then after the aggregator has successfully done this, we'll clear uncommitted
        self.uncommitted = []



    def train(self, t_round):
        """
        function to ask the user to train based on their local data
        input: - round: A Round, with a training function that the user will do,
                        and a data function that gets any subset of data from the user
                        to participate in the data
        output: whatever is required of the user from the aggregator, usually weights, to
                update the global model
        """
        # get the training function and data function
        train_f, data_f, rid = t_round.get_training_function(), t_round.get_data_function(), t_round.get_round_id()
        # getting the data to be trained
        filtered_data = self.get_opted_in_data()
        training_data = data_f(filtered_data)
        # this part is to update the training_data that it has already participated in t_round
        for ele in training_data:
            assert type(ele) == dict
            if "rids" not in ele: ele["rids"] = []
            ele["rids"].append(rid)
        # getting the weights from the training function
        output = train_f(training_data)
        # # ask the logger to log the round id
        # self.logger.log_round_participated(uid, rid, output) -- this should probably happen at the aggregator level
        # return the data so the aggregator can get it
        return output

    
    def update_weights(self, prev_rid, new_rid, update_func):
        try:
            assert prev_rid in self.rid_to_local_weight
            old_weight = self.rid_to_local_weight[prev_rid]
            new_weight = update_func(old_weight)
            self.update_rid_to_local_weight(new_rid, new_weight)
        except Exception as e:
            print(e)
            return False
        return True




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



class Aggregator:
    def __init__(self, logger, compliance_mode=VERY_WEAK):
        self.logger = logger
        self.compliance_mode = compliance_mode
        pass


    def remove_user_from_rounds(self, uid: int, rids: list, compliance_mode=VERY_WEAK):
        """
        function to remove the user from round (i.e. their weight contribution in the round). List of things that its doing:
        - identify what the smallest round id to remove is
        - for each round from the smallest round id to the end:
            + get the new weights WITHOUT those that this user contributed
            + get the training plan (what the aggregation function is)
            + run the aggeregation plan on the previous training plan and the new weights of this round
            + by ^, getting (1) the updated global weights, and then (2) the WEIGHT UPDATES TO individual devices that
            participated in the training
            + updating the log on what the resulting global weight checkpoint is after this
            + sending the weight updates to the devices that are participating

        return:
        - true if successful
        - false if not
        """
        min_rid = min(rids)
        # and then cascadedly update the weights
        for rid in range(min_rid, self.logger.get_next_rid()):
            try:
                # getting the weights contributed by others except for this uid
                new_weights = self.logger.weights_given_rid_excluding_uids(rid, excluding_uids=[uid])
                # getting the round and the corresponding aggregation function
                t_round = self.logger.get_round(rid)
                aggregator = t_round.get_aggregation_function()
                prev_weights = self.logger.get_global_checkpoint(rid - 1) # none or the checkpoints from previous
                # train again
                updated_weights, uid_to_local_weights = aggregator(weights=new_weights, prev_weight=prev_weights)
                # and then update the global checkpoint of this
                self.logger.set_global_checkpoint(rid, updated_weights, replace=True)
                # after we have successfully updated the global checkpoint without an issue, we will delete the user
                # from participation of this round
                self.logger.delete_round_participated(uid, rid)
                # and then send the weight updatesto devices that are participating
                for uid_to_locally_update in uid_to_local_weights:
                    update_function = uid_to_local_weights[uid_to_locally_update]
                    user_to_update = self.logger.get_user(uid_to_locally_update)
                    # tell the user to update with this update function
                    self.user_to_update.update_weights(rid-1, rid, update_function)
            except Exception as e:
                # This is potentially where the weaker and harder forms of compliance would happen --
                # return False if fails to delete, vs. continue and just skipping for those that are too hard
                print("Exception caught: ", e)
                if compliance_mode == STRICT: return False
                continue
        return True


    def update_user_participation_in_rounds(self, uid: int, rids: list, compliance_mode=VERY_WEAK):
        """
        function to update user's participation in different rounds (i.e. update the weights in the round).

        The way that the updates are happening is that its based on local changes (i.e. user cannot update their
        weights directly (input the weights in), but they update the data on their phone and call update user participation
        and that would update based on their local data). This can be a lot more useful with the user being able
        to identify which part of their data is involved in training

        list of things that its doing:
        - find the minimum round that we are updating
        - and then casecadedly update the weights. specifically, at each round:
        """
        min_rid = min(rids)
        for rid in range(min_rid, self.logger.get_next_rid()):
            try:
                # get the round
                t_round = self.logger.get_round(rid)
                # get the user
                user = self.logger.get_user(uid)
                # get the output from training
                output = user.train(t_round)
                # get the weight contributed by this uid to this rid in the past
                old_weight = self.logger.get_weight_contributed_by_device(uid, rid)
                # compare // TODO: Does this method of comparison work lmao and what should we use the type of output/old_weight
                if output == old_weight: continue
                else:
                    # in this case we would get the aggregation function, and train over again
                    aggregation_f = t_round.get_aggregation_function()
                    previous_global_checkpoint = self.logger.get_global_checkpoint(rid-1)
                    # construct a new set of weights returned for this round
                    new_weights_returned = {}
                    new_weights_returned[uid] = output
                    uids = self.logger.get_uids_from_rid(rid)
                    for other_uid in uids:
                        if other_uid != uid:
                            new_weights_returned[other_uid] = self.logger.get_weight_contributed_by_device(other_uid, rid)
                    # use the aggregation function to get the global weights and to get the updates on each user's device
                    global_weights, uid_to_local_weight_fs = aggregation_f(weights=new_weights_returned, prev_weight=previous_global_checkpoint)
                    # replace with what we currently have
                    self.logger.set_global_checkpoint(rid, global_weights, replace=True)
                    # ask each user to update the weights
                    for uid_to_locally_update in uid_to_local_weight_fs:
                        # user:
                        user = self.logger.get_user(uid_to_locally_update)
                        # update_function: function that would take in a previous set of weights and output a new set of weights
                        # should handle input=None (in case this is the first training round)
                        update_function = uid_to_local_weight_fs[uid_to_locally_update]
                        # tell the user to update with this update function
                        self.user.update_weights(rid-1, rid, update_function)
            except Exception as e:
                if compliance_mode == STRICT: return False
                continue
        return True


    def basic_train(self, t_round, num_participants):
        """
        function to do the server's training algorithm as demonstrated in the paper. The list of things that it will do:
        - Select randomly num_participants users to train stuff
        - For each of these, call the train() function on each user's device
        """
        # get the training function and the data selection function for each user
        try:
            train_f, data_f, rid = t_round.get_training_function(), t_round.get_data_function(),t_round.get_round_id()
            selected_users = self.logger.sample_users(num_participants)
            weights_returned = {}
            for uid in selected_users:
                user = self.logger.get_user(uid)
                # tell that user to train and give back the weights
                output = user.train(t_round) # this should already log their weight contribution
                # update to the weights_returned
                weights_returned[uid] = output
            # get the aggregator and the last checkpoint
            previous_global_checkpoint = self.logger.get_global_checkpoint(rid - 1)
            aggregation_f = t_round.get_aggregation_function()
            # call aggregation_f to get the global weight updates and the weight updates function to send back to devices
            global_weights, uid_to_local_weight_fs = aggregation_f(weights=weights_returned, prev_weight=previous_global_checkpoint)
            # set the global checkpoint (so the weights, not the updates)
            self.logger.set_global_checkpoint(rid, rid, global_weights)
            # ask users to update the weights
            for uid_to_locally_update in uid_to_local_weight_fs:
                # user:
                user = self.logger.get_user(uid_to_locally_update)
                # update_function: function that would take in a previous set of weights and output a new set of weights
                # should handle input=None (in case this is the first training round)
                update_function = uid_to_local_weight_fs[uid_to_locally_update]
                # tell the user to update with this update function
                self.user.update_weights(rid-1, rid, update_function)
            return True
        except Exception as e:
            print("Exception caught: ", e)
            return False
            
            
        



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
