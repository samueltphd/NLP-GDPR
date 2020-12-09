from flask import render_template, redirect, request, Flask
import sys
import pickle
import random
from model.income import import_census, loss
from model.aggregator import Aggregator

app = Flask(__name__)

######################## DATASET STUFF FOR EVALUATION ####################################
CENSUS_FILE_PATH = "./income_data.csv"
xtrain, ytrain, xtest, ytest = import_census(CENSUS_FILE_PATH)

def randomly_optin_data(uid_to_user_dict):
    for temp_uid in uid_to_user_dict:
        datapoints = uid_to_user_dict[temp_uid].data
        random_percentage_opted_in = float(random.randrange(50, 100))
        for data in datapoints:
            rand_this = float(random.randrange(100))
            if rand_this <= random_percentage_opted_in: data['opt_in'] = True

##########################################################################################

INCOME_PICKLE_PATH = "income-logger.pkl"
with open(INCOME_PICKLE_PATH, "rb") as pickle_file:
    income_logger = pickle.load(pickle_file)
    for uid in income_logger.uid_to_user:
        user = income_logger.uid_to_user[uid]
        user.uncommitted_delete = []
        user.uncommitted_update = []

# randomly opt in data
randomly_optin_data(income_logger.uid_to_user)
NO_COMPLIANCE, NEUTRAL, STRONG, STRICT = 0, 1, 2, 3
COMPLIANCE_MODE_DICT = {
    "NO_COMPLIANCE": NO_COMPLIANCE,
    "NEUTRAL": NEUTRAL,
    "STRONG": STRONG,
    "STRICT": STRICT
}
current_compliance_mode_str = "NEUTRAL"
current_compliance_mode = COMPLIANCE_MODE_DICT[current_compliance_mode_str]

income_aggregator = Aggregator(income_logger, current_compliance_mode)

current_uid = 0

@app.route("/")
def home():
    income_aggregated_info_dict = get_aggregated_information_about_logger(income_logger)
    print("Global compliance mode str: ", current_compliance_mode_str)
    print("Global compliance mode: ", current_compliance_mode)
    return render_template("index.html", incomelogger=income_logger, mnistlogger=None, user=income_logger.get_user(current_uid),
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()),
    incomeAggregated=income_aggregated_info_dict)


@app.route("/income/<uid>")
def income_user_view(uid):
    """
    Function to display data from a user in the income context
    """
    uid = int(uid)
    user = income_logger.uid_to_user[uid]
    aggregated_info = get_aggregated_info_about_user(user, income_logger)
    return render_template("income.html", user=user, aggregatedUserInfo=aggregated_info,
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()))


@app.route("/change_permission/<user_id>/<data_id>")
def change_data_permission(user_id, data_id):
    """
    function to toggle the data permission
    """
    uid, data_id = int(user_id), int(data_id)
    user = income_logger.uid_to_user[uid]
    datapoints = user.data
    for point in datapoints:
        if int(point['id']) != data_id: continue
        user.change_data_permission(data_id, value=not(point['opt_in']), deep=current_compliance_mode >= 2)
    aggregated_info = get_aggregated_info_about_user(user, income_logger)
    return render_template("income.html", user=user, aggregatedUserInfo=aggregated_info,
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()))


@app.route("/delete_data_local/<user_id>/<data_id>")
def perform_local_delete(user_id, data_id):
    uid, data_id = int(user_id), int(data_id)
    user = income_logger.uid_to_user[uid]
    user.remove_data(data_id, deep=False)
    aggregated_info = get_aggregated_info_about_user(user, income_logger)
    return render_template("income.html", user=user, aggregatedUserInfo=aggregated_info,
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()))


@app.route("/delete_data_deep/<user_id>/<data_id>")
def perform_deep_delete(user_id, data_id):
    uid, data_id = int(user_id), int(data_id)
    user = income_logger.uid_to_user[uid]
    user.remove_data(data_id, deep=True)
    aggregated_info = get_aggregated_info_about_user(user, income_logger)
    return render_template("income.html", user=user, aggregatedUserInfo=aggregated_info,
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()))
    

@app.route("/update_data_local/<user_id>/<data_id>")
def perform_local_update(user_id, data_id):
    uid, data_id = int(user_id), int(data_id)
    user = income_logger.uid_to_user[uid]
    for d in user.data:
        if d["id"] == data_id:
            new_data = d.copy()
    user.update_data(data_id, new_data, deep=False)
    aggregated_info = get_aggregated_info_about_user(user, income_logger)
    return render_template("income.html", user=user, aggregatedUserInfo=aggregated_info,
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()))


@app.route("/commit_deletes_and_updates/<user_id>")
def commit_deletes_and_updates(user_id):
    uid = int(user_id)
    user = income_logger.uid_to_user[uid]
    user.request_aggregator_update(income_aggregator)
    aggregated_info = get_aggregated_info_about_user(user, income_logger)
    return render_template("income.html", user=user, aggregatedUserInfo=aggregated_info,
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()))


@app.route("/update_data_deep/<user_id>/<data_id>")
def perform_deep_update(user_id, data_id):
    uid, data_id = int(user_id), int(data_id)
    user = income_logger.uid_to_user[uid]
    for d in user.data:
        if d["id"] == data_id:
            new_data = d.copy()
    user.update_data(data_id, new_data, deep=True)
    aggregated_info = get_aggregated_info_about_user(user, income_logger)
    return render_template("income.html", user=user, aggregatedUserInfo=aggregated_info,
    complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()))


def get_current_compliance_mode_str():
    return current_compliance_mode_str

def get_current_compliance_mode():
    return current_compliance_mode

@app.route("/server")
def display_server():
    """
    function to display the server side of things
    """
    print("Global compliance mode str - /server: ", current_compliance_mode_str)
    print("Global compliance mode - /server: ", current_compliance_mode)
    return render_template("server.html", server=income_aggregator, incomeLogger=income_logger, user=income_logger.get_user(current_uid),
    complianceMode=get_current_compliance_mode_str(), uids=list(sorted(income_logger.uid_to_rids.keys())),
    complianceOptions=list(COMPLIANCE_MODE_DICT.keys()), queueLen=len(income_aggregator.urm.values))
    

@app.route("/changeCompliance", methods=["POST"])
def changeCompliance():
    print("Received change compliance request")
    global current_compliance_mode_str, current_compliance_mode
    complianceModeStr = request.form['selectedCompliance']
    current_compliance_mode_str = complianceModeStr
    current_compliance_mode = COMPLIANCE_MODE_DICT[current_compliance_mode_str]
    income_aggregator.compliance_mode = COMPLIANCE_MODE_DICT[current_compliance_mode_str]
    income_aggregated_info_dict = get_aggregated_information_about_logger(income_logger)
    return redirect("/")
    # return render_template("index.html", incomelogger=income_logger, mnistlogger=None, user=income_logger.get_user(current_uid),
    # complianceMode=current_compliance_mode_str, uids=list(sorted(income_logger.uid_to_rids.keys())), complianceOptions=list(COMPLIANCE_MODE_DICT.keys()),
    # incomeAggregated=income_aggregated_info_dict)


@app.route("/changeUid", methods=["POST"])
def changeUid():
    global current_uid
    newUid = int(request.form['uid-selector'])
    current_uid = newUid
    income_aggregated_info_dict = get_aggregated_information_about_logger(income_logger)
    return redirect("/")


@app.route("/process_batch_request")
def process_batch_request():
    income_aggregator.process_badge_request()
    return render_template("server.html", server=income_aggregator, incomeLogger=income_logger, user=income_logger.get_user(current_uid),
    complianceMode=get_current_compliance_mode_str(), uids=list(sorted(income_logger.uid_to_rids.keys())),
    complianceOptions=list(COMPLIANCE_MODE_DICT.keys()), queueLen=len(income_aggregator.urm.values))


########################### We temporarily don't care about these ########################
@app.route("/mnist/<uid>")
def mnist_user_view(uid):
    """
    Function to display data from a user in the mnist context
    """
    uid = int(uid)
    user = income_logger.uid_to_user[uid]
    return render_template('mnist.html', user=user)
    

#################################### HELPER FUNCTIONS ####################################
def get_aggregated_info_about_user(user, logger):
    stats = {}
    uid = user.uid
    user.data = sorted(user.data, key=lambda x: x['id'])
    for d in user.data:
        d["rids"] = list(set(d["rids"]))
    # uncommitted delete
    stats["num_uncommitted_delete"] = len(user.uncommitted_delete)
    stats["num_uncommitted_update"] = len(user.uncommitted_update)
    # num rounds participated
    stats["num_rounds_participated"] = len(logger.uid_to_rids[uid])
    # num datapoints:
    stats["num_datapoints"] = len(user.data)
    # num datapoints opted in
    stats["num_optedin_datapoints"] = len(user.get_opted_in_data())
    return stats


def get_aggregated_information_about_logger(logger):
    stats = {}
    # number of users participated
    stats["num_users_participated"] = len(logger.uid_to_user.keys())
    # number of rounds participated per user
    min_round_participated, average_round_participated, max_round_participated = logger.next_rid + 1, 0, 0
    for u in logger.uid_to_rids:
        num_rounds_participated = len(logger.uid_to_rids[u])
        min_round_participated = min(min_round_participated, num_rounds_participated)
        average_round_participated += num_rounds_participated
        max_round_participated = max(max_round_participated, num_rounds_participated)
    average_round_participated = average_round_participated/stats["num_users_participated"]
    stats["average_num_rounds_partipated_per_user"] = average_round_participated
    stats["min_num_rounds_participated_per_user"] = min_round_participated
    stats["max_num_rounds_participated_per_user"] = max_round_participated
    # how many datapoints does each user have
    min_datapoints_per_user, max_datapoints_per_user, average_datapoints_per_user = 999999999, 0, 0
    for u in logger.uid_to_user:
        temp_user = logger.uid_to_user[u]
        num_datapoints = len(temp_user.data)
        min_datapoints_per_user = min(min_datapoints_per_user, num_datapoints)
        max_datapoints_per_user = max(max_datapoints_per_user, num_datapoints)
        average_datapoints_per_user += num_datapoints
    average_datapoints_per_user = average_datapoints_per_user/len(logger.uid_to_user.keys())
    stats["average_num_datapoints_per_user"] = average_datapoints_per_user
    stats["max_num_datapoints_per_user"] = max_datapoints_per_user
    stats["min_num_datapoints_per_user"] = min_datapoints_per_user
    # how many *opted in* datapoints does each user have
    min_datapoints_optedin_per_user, max_datapoints_optedin_per_user, average_datapoints_optedin_per_user = 999999999, 0, 0
    for u in logger.uid_to_user:
        temp_user = logger.uid_to_user[u]
        num_opted_in_datapoints = len(temp_user.get_opted_in_data())
        min_datapoints_optedin_per_user = min(min_datapoints_optedin_per_user, num_opted_in_datapoints)
        max_datapoints_optedin_per_user = max(max_datapoints_optedin_per_user, num_opted_in_datapoints)
        average_datapoints_optedin_per_user += num_opted_in_datapoints
    average_datapoints_optedin_per_user = average_datapoints_optedin_per_user / len(logger.uid_to_user.keys())
    stats["average_datapoints_optedin_per_user"] = average_datapoints_optedin_per_user
    stats["max_num_datapoints_optedin_per_user"] = max_datapoints_optedin_per_user
    stats["min_num_datapoints_optedin_per_user"] = min_datapoints_optedin_per_user
    # how many rounds
    stats["num_rounds"] = len(logger.rid_to_global_checkpoints.keys()) - 1
    # how many users does each round have
    min_users_per_round, avg_users_per_round, max_users_per_round = stats["num_users_participated"] + 1, 0, 0
    for r in logger.rid_to_uids:
        num_temp_users = len(logger.rid_to_uids[r])
        min_users_per_round = min(min_users_per_round, num_temp_users)
        max_users_per_round = max(max_users_per_round, num_temp_users)
        avg_users_per_round += num_temp_users
    avg_users_per_round = avg_users_per_round/len(logger.rid_to_uids.keys())
    stats["average_num_users_per_round"] = avg_users_per_round
    stats["max_num_users_per_round"] = max_users_per_round
    stats["min_num_users_per_round"] = min_users_per_round
    # also doing this hacking - to be deleted
    # for u in logger.uid_to_rids:
    #     temp_user = logger.uid_to_user[u]
    #     # rounds participated:
    #     rids_participated_given_u = logger.uid_to_rids[u]
    #     num_rounds_participated_per_user = len(rids_participated_given_u)
    #     # for each datapont, now we're going to update the rounds participated
    #     for datapoint in temp_u.data:
    #         # randint number of rounds that this has participated in
    #         num_rounds_participating_per_datapoint = random.randrange(num_rounds_participated)
    #         # and then for each rids, 
            
    # loss_dict = {}
    # loss per each round
    # for r in logger.rid_to_global_checkpoints:
    #     r_weight = logger.rid_to_global_checkpoints[r]
    #     r_loss = loss(xtest, ytest, r_weight)
    #     loss_dict[r] = r_loss
    # print(loss_dict)
    return stats



if __name__ == "__main__":
    try:
        app.run(debug=True)
    except OSError:
        e = 'Cannot run multiple instances of app at same time'
        e += '\nStop all other app.py instances and try again.'
        raise OSError(e)


