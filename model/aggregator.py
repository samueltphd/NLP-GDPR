# different modes of GDPR compliance
NO_COMPLIANCE, NEUTRAL, STRONG, STRICT = 0, 1, 2, 3
DELETE, UPDATE = "DELETE", "UPDATE"

# SHALLOW DELETION/UPDATES: NEUTRAL
# BADGE DELETION/UPDATES: STRONG
# ONE USER AT A TIME DELETION/UPDATES: STRICT
class Aggregator:
    def __init__(self, logger, compliance_mode=NO_COMPLIANCE, badge_limit=10):
        self.logger = logger
        self.compliance_mode = compliance_mode
        if compliance_mode == STRONG:
            # data structure to map from id (int) to list of (round, request_type)
            self.to_updates = {}
            self.badge_limit = badge_limit

    
    def get_to_update_dict(self):
        return self.to_updates


    def user_request_update_weak(self, uid: int, rids: list, request_type=DELETE):
        assert self.compliance_mode == NEUTRAL
        try:
            # What this will do is just call the logger to remove user information in there
            for rid in rids:
                # if type == DELETE or UPDATE, both has to go through this
                self.logger.delete_round_participated(uid, rid)
                if request_type == UPDATE:
                    # get the round
                    t_round = self.logger.get_round(rid)
                    # get the user
                    user = self.logger.get_user(uid)
                    # get the output from training
                    output = user.train(t_round)
                    # get the weight contributed by this uid to this rid in the past
                    old_weight = self.logger.get_weight_contributed_by_device(uid, rid)
                    # if its the same, then there's no need to update anything this round
                    if output == old_weight: continue
                    # else, we update stuff
                    else: self.logger.log_round_participated(uid, rid, output)
        except Exception as e:
            print("Exception caught: ", e)
            return False
        return True

    
    def user_request_update_strong(self, uid: int, rids: list, request_type=DELETE):
        assert self.compliance_mode == STRONG
        try:
            # for each round id
            for rid in rids:
                # just add the update request to our self.to_updates data structure
                if uid not in self.to_updates: self.to_updates[uid] = []
                if (rid, request_type) not in self.to_updates[uid]:
                    self.to_updates[uid].append((rid, request_type))
            # if the length exceeds a certain threshold:
            if len(rid) >= self.badge_limit:
                self.process_badge_request()
            # if all is done, return True (successful)
            return True
        # if somewhere fails, return False
        return False



    def process_badge_request(self):
        assert self.compliance_mode == STRONG
        to_update_dict = self.get_to_update_dict()
        # temp variables to keep track of the rid -> uids that requested to delete/update
        rid_to_uids_delete, rid_to_uids_update = {}, {}
        temp = {DELETE: rid_to_uids_delete, UPDATE: rid_to_uids_update}
        # for each user in to_update_dict:
        for uid in to_update_dict:
            # there will be an associated list of rounds that this user requested to update
            rids_to_types = to_update_dict[uid]
            for rid_type in rids_to_types:
                rid, t = rid_type[0], rid_type[1]
                # get the rid_to_uid_delete/update dict to update depending on the type of request
                rid_to_uids_type = temp[t]
                # and then check if this rid is already in there and stuff
                if rid not in rid_to_uids_type: rid_to_uids_type[rid] = []
                # associate this uid with this round update
                if uid not in rid_to_uids_type[rid]: rid_to_uids_type[rid].append(uid)
        # okay. now we gonna process the delete dicts first - they easier
        for rid in rid_to_uids_delete:
                
                

            
        


    def user_request_update_strict(self, uid: int, rids: list, request_type=DELETE):
        assert self.compliance_mode == STRICT
        try:
            # get the user
            user = self.logger.get_user(uid)
            # get the round id to start everything with
            min_rid = min(rids)
            # and then cascadedly update the weights
            for rid in range(min_rid, self.logger.get_next_rid()):
                # get the round
                t_round = self.logger.get_round(rid)
                # get the output from training
                output = user.train(t_round)
                # get the weight contributed by this uid to this rid in the past
                old_weight = self.logger.get_weight_contributed_by_device(uid, rid)
                # get all the weights from previous rounds excluding this uid
                new_weights = self.logger.weights_given_rid_excluding_uids(rid, excluding_uids=[uid])
                aggregator = t_round.get_aggregation_function()
                prev_weights = self.logger.get_global_checkpoint(rid - 1) # none or the checkpoints from previous
                # if update, then additionally you would update the new_weights to include the new one
                if request_type == UPDATE:
                    new_weights[uid] = output
                # and then using the aggregator function to get the new global weights and the function to update each
                updated_weights, uid_to_local_weights = aggregator(weights=new_weights, prev_weight=prev_weights)
        except Exception as e:
            print("Exception caught in user_request_update_strong: ", e)
            return False
        return True



    def user_request_update(self, uid, rids: list, request_type=DELETE, compliance_mode=NO_COMPLIANCE):
        # if it is a weak compliance mode
        if compliance_mode == NEUTRAL: self.user_request_update_weak(uid, rids, request_type)
        if compliance_mode == STRONG: self.user_request_update_strong(uid, rids, request_type)
        if compliance_mode == STRICT: self.user_request_update_strict(uid, rids, request_type)



    def remove_user_from_rounds(self, uid: int, rids: list, compliance_mode=NO_COMPLIANCE):
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


    def update_user_participation_in_rounds(self, uid: int, rids: list, compliance_mode=NO_COMPLIANCE):
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
                        user.update_weights(rid-1, rid, update_function)
            except Exception as e:
                if compliance_mode == STRICT: return False
                continue
        return True


    def basic_train(self, t_round):
        print(t_round)
        """
        function to do the server's training algorithm as demonstrated in the paper. The list of things that it will do:
        - Select randomly num_participants users to train stuff
        - For each of these, call the train() function on each user's device
        """
        # get the training function and the data selection function for each user
        try:
            train_f, rid = t_round.get_training_function(), t_round.get_round_id()
            selected_users = self.logger.sample_users(t_round.num_participants)
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
                user.update_weights(rid-1, rid, update_function)
            return True
        except Exception as e:
            print("Exception caught: ", e)
            return False
