# different modes of GDPR compliance
NO_COMPLIANCE, NEUTRAL, STRONG, STRICT = 0, 1, 2, 3
DELETE, UPDATE = "DELETE", "UPDATE"

# SHALLOW DELETION/UPDATES: NEUTRAL
# BADGE DELETION/UPDATES: STRONG
# ONE USER AT A TIME DELETION/UPDATES: STRICT

class User:
    def __init__(self, uid, aggregator, logger, compliance_mode=NO_COMPLIANCE):
        self.uid = uid # user id
        # self.aggregator = aggregator # aggregator that this user communicates with
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
        # these data structures are to keep track of local changes uncommitted to the global level
        # store all the round ids that are uncommitted
        self.uncommitted_delete = []
        self.uncommitted_update = []


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
            self.uncommitted_delete.extend(data_point["rids"])
            self.uncommitted_delete = list(set(self.uncommitted_delete))


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
            self.uncommitted_update.extend(data_point["rids"])
            self.uncommitted_update = list(set(self.uncommitted_update))


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
            self.uncommitted_delete.extend(data_point["rids"])
            self.uncommitted_delete = list(set(self.uncommitted_delete))


    def request_aggregator_update(self, aggregator):
        aggregator.urm.add_request([(True, self.uid, v) for v in self.uncommitted_update])
        aggregator.urm.add_request([(True, self.uid, v) for v in self.uncommitted_delete])
        self.uncommitted_delete = []
        self.uncommitted_update = []




    def train(self, t_round, global_weights):
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
        output, localLoss = train_f(training_data, global_weights)
        # # ask the logger to log the round id
        # self.logger.log_round_participated(uid, rid, output) -- this should probably happen at the aggregator level
        # TODO: Ask the user to log that these pieces of data participated in the training
        for datapoint in training_data:
            d_id = datapoint["id"]
            # associate did -> rid
            if d_id not in self.data_id_to_rids: self.data_id_to_rids[d_id] = []
            self.data_id_to_rids[d_id].append(rid)
            # associate rid -> did
            if rid not in self.rid_to_data_ids: self.rid_to_data_ids[rid] = []
            self.rid_to_data_ids[rid].append(d_id)
            # adding this data id

        # return the data so the aggregator can get it
        return output, localLoss


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
