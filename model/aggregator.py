import threading
import time
from copy import deepcopy
import math

# different modes of GDPR compliance
NO_COMPLIANCE, NEUTRAL, STRONG, STRICT = 0, 1, 2, 3
DELETE, UPDATE = "DELETE", "UPDATE"

# SHALLOW DELETION/UPDATES: NEUTRAL
# BADGE DELETION/UPDATES: STRONG
# ONE USER AT A TIME DELETION/UPDATES: STRICT
class UserRequestManager:
    def __init__(self, agg, compliance_mode):
        self.agg = agg
        self.values = []
        self.lock = threading.Lock()
        self.compliance_mode = compliance_mode

    def add_request(self, value):
        self.lock.acquire()
        for v in value:
            self.values.append(v)
        self.lock.release()

    def handle_requests(self):
        batch_size = 0

        if self.compliance_mode == 0:
            batch_size = math.inf
        elif self.compliance_mode == 1:
            batch_size = 10000
        elif self.compliance_mode == 2:
            batch_size = 100

        self.lock.acquire()
        if len(self.values) < batch_size:
            self.lock.release()
            return False
        else:
            for v in self.values:
                self.agg.user_request_update(v[1], v[2])
            self.lock.release()
            return True

class Aggregator:
    def __init__(self, logger, compliance_mode=NO_COMPLIANCE, badge_limit=0):
        self.logger = logger
        self.compliance_mode = compliance_mode
        self.lock = threading.Lock()
        self.urm = UserRequestManager(self, compliance_mode)
        self.producer_qs = None
        self.consumer_qs = None
        if compliance_mode == STRONG:
            # data structure to map from id (int) to list of (round, request_type)
            self.to_updates = {}
            self.badge_limit = badge_limit


    def get_to_update_dict(self):
        return self.to_updates

    def clear_to_update_dict(self):
        self.to_updates = {}


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
                    self.producer_qs[uid].enque((user.train, t_round, self.logger.get_global_checkpoint(rid)))

                    result = self.consumer_qs[uid].deque()
                    while result == None:
                        time.sleep(3)
                        result = self.consumer_qs[uid].deque()

                    output, localLoss = result
                    # update the state
                    self.logger.log_round_participated(uid, rid, output)
                    # dont do anything with local loss since we are not retraining again
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
        except Exception as e:
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
        min_rid = min([min(rid_to_uids_delete), min(rid_to_uids_update)])
        # starting from that rid onwards
        try:
            # variable to keep track of the loss
            loss_train = []
            for rid in range(min_rid, self.logger.get_next_rid()):
                # get the round
                t_round = self.logger.get_round(rid)
                # get the output from training
                self.producer_qs[uid].enque((user.train, t_round, self.logger.get_global_checkpoint(rid)))

                result = self.consumer_qs[uid].deque()
                while result == None:
                    time.sleep(3)
                    result = self.consumer_qs[uid].deque()

                output, localLoss = result
                # if there is at least some user that requested their participation to
                # be deleted from this round
                if rid in rid_to_uids_delete: excluding = rid_to_uids_delete[rid]
                else: excluding = []
                # get the new weights to aggregate in this round
                new_weights = self.logger.weights_given_rid_excluding_uids(rid, excluding_uids=excluding)
                # if there is at least some user who requested their contribution to be updated:
                if rid in rid_to_uids_update:
                    # for each user who did so
                    loss_locals = []
                    for uid in rid_to_uids_update[rid]:
                        # we will get the output from training on that user device
                        self.producer_qs[uid].enque((user.train, t_round, self.logger.get_global_checkpoint(rid)))

                        result = self.consumer_qs[uid].deque()
                        while result == None:
                            time.sleep(3)
                            result = self.consumer_qs[uid].deque()

                        output, localLoss = result
                        print("Local loss: ", localLoss)
                        # update the new_weights to reflect the (potentially) new contribution
                        # from this uid
                        if uid in new_weights: # to prevent the case where a deletion was scheduled -
                            new_weights[uid] = output
                        # append loss locals
                        loss_locals.append(deepcopy(localLoss))
                    print("loss_locals: ", loss_locals)
                    loss_avg = sum(loss_locals)/len(loss_locals)
                    print('Round {:3d}, Average loss {:,3f}'.format(rid, loss_avg))
                # by this time, we have already had a finalized new_weights. Now we will aggregate!
                aggregator = t_round.get_aggregation_function()
                prev_weights = self.logger.get_global_checkpoint(rid - 1)
                # aggregate the different ids
                updated_weights, uid_to_local_weights = aggregator(weights=new_weights, prev_weight=prev_weights)
                # send the weight updates to each device
                for uid_to_locally_update in uid_to_local_weights:
                    update_function = uid_to_local_weights[uid_to_locally_update]
                    user_to_update = self.logger.get_user(uid_to_locally_update)
                    # tell the user to update with this update function
                    user_to_update.update_weights(rid-1, rid, update_function)
                # and then afterwards, update the global weights
                self.logger.set_global_checkpoint(rid, updated_weights, replace=True)
                # after the global model has already been updated, looping through the new_weights to update the
                # weight contribution from each user
                for uid in new_weights:
                    # update this uid participation in the round
                    self.logger.log_round_participated(uid, rid, new_weights[uid])
            # and then, when we are done with everything, we will empty self.to_updates
            self.clear_to_update_dict()
        except Exception as e:
            print("Exception caught: ", e)
            return False
        return True



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
                self.producer_qs[uid].enque((user.train, t_round, self.logger.get_global_checkpoint(rid)))

                result = self.consumer_qs[uid].deque()
                while result == None:
                    time.sleep(3)
                    result = self.consumer_qs[uid].deque()

                output, localLoss = result
                # # get the weight contributed by this uid to this rid in the past
                # old_weight = self.logger.get_weight_contributed_by_device(uid, rid)
                # # if the old_weight contributed by the device is the same as the new weight contribution, it
                # # means that the update has "converged", and so we just break and return
                # if old_weight == output:
                #     break
                # get all the weights from previous rounds excluding this uid
                new_weights = self.logger.weights_given_rid_excluding_uids(rid, excluding_uids=[uid])
                aggregator = t_round.get_aggregation_function()
                prev_weights = self.logger.get_global_checkpoint(rid - 1) # none or the checkpoints from previous
                # if update, then additionally you would update the new_weights to include the new one
                if request_type == UPDATE:
                    new_weights[uid] = output
                # and then using the aggregator function to get the new global weights and the function to update each
                updated_weights, uid_to_local_weights = aggregator(weights=new_weights, prev_weight=prev_weights)
                # send the weight updates to each device
                for uid_to_locally_update in uid_to_local_weights:
                    update_function = uid_to_local_weights[uid_to_locally_update]
                    user_to_update = self.logger.get_user(uid_to_locally_update)
                    # tell the user to update with this update function
                    user_to_update.update_weights(rid-1, rid, update_function)
                # and then afterwards, update the global weights
                self.logger.set_global_checkpoint(rid, updated_weights, replace=True)
                # and then update this uid's participation in the logger
                self.logger.log_round_participated(uid, rid, output)
        except Exception as e:
            print("Exception caught in user_request_update_strong: ", e)
            return False
        return True


    def user_request_update(self, uid, rids: list, request_type=DELETE, compliance_mode=NO_COMPLIANCE):
        # if it is a weak compliance mode
        self.lock.acquire()
        if compliance_mode == NEUTRAL: self.user_request_update_weak(uid, rids, request_type)
        if compliance_mode == STRONG: self.user_request_update_strong(uid, rids, request_type)
        if compliance_mode == STRICT: self.user_request_update_strict(uid, rids, request_type)
        self.lock.release()


    def basic_train(self, t_round, producer_qs, consumer_qs, train_time=1):
        """
        function to do the server's training algorithm as demonstrated in the paper. The list of things that it will do:
        - Select randomly num_participants users to train stuff
        - For each of these, call the train() function on each user's device
        """
        self.producer_qs = producer_qs
        self.consumer_qs = consumer_qs
        # get the training function and the data selection function for each user
        # try:
        rid = t_round.get_round_id()
        # for ep in range(t_round.get_epochs()):
        for ep in range(1):
            print("Epoch: ", ep)
            if ep == 0: previous_global_checkpoint = self.logger.get_global_checkpoint(rid - 1)
            else: previous_global_checkpoint = self.logger.get_global_checkpoint(rid)
            selected_users = self.logger.sample_users(t_round.num_participants)
            weights_returned = {}
            for uid in selected_users:
                user = self.logger.get_user(uid)
                # tell that user to train and give back the weights
                producer_qs[uid].enque((user.train, t_round, previous_global_checkpoint))
                print("Training request sent")
            # begin retreiving user weights from the users
            received = 0
            loss_locals = []
            while received < len(selected_users):
                for uid in selected_users:
                    user = self.logger.get_user(uid)
                    # the user will place their weights in their producer queue, so
                    # try to deque from that queue if it isn't empty
                    output = consumer_qs[uid].deque()
                    if output is not None:
                        print("Aggregator received weight from user " + str(uid))
                        # parse the result
                        output, localLoss = output
                        print("local loss: ", localLoss)
                        loss_locals.append(deepcopy(localLoss))
                        # update to the weights_returned
                        weights_returned[uid] = deepcopy(output)
                        received += 1
                if received == 0:
                    # wait for the users to train on their data
                    time.sleep(train_time)
                    for q in producer_qs:
                        if q.deque() is not None:
                            self.logger.set_global_checkpoint(rid, previous_global_checkpoint, replace=True)
                            return False
            print("Loss locals: ", loss_locals)
            loss_avg = sum(loss_locals)/len(loss_locals)
            print('Epoch {:3d}, Round {:3d}, Average loss {:.3f}'.format(ep, rid, loss_avg))
            # get the aggregator and the last checkpoint
            aggregation_f = t_round.get_aggregation_function()
            # call aggregation_f to get the global weight updates and the weight updates function to send back to devices
            global_weights, uid_to_local_weight_fs = aggregation_f(uid_to_weights=weights_returned, prev_weights=previous_global_checkpoint)
            # ask users to update the weights
            for uid_to_locally_update in uid_to_local_weight_fs:
                # user:
                user = self.logger.get_user(uid_to_locally_update)
                # update_function: function that would take in a previous set of weights and output a new set of weights
                # should handle input=None (in case this is the first training round)
                update_function = uid_to_local_weight_fs[uid_to_locally_update]
                # tell the user to update with this update function
                if user is not None:
                    user.update_weights(rid-1, rid, update_function)
            # set the global checkpoint (so the weights, not the updates)
            self.logger.set_global_checkpoint(rid, global_weights, replace=True)
            # and then, got hrough each user to update their weight contribution in this round
            for uid in weights_returned:
                self.logger.log_round_participated(uid, rid, weights_returned[uid])
        return True
        # except Exception as e:
        #     print("Exception caught: ", e)
        #     return False
