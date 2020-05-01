### TO DO:
#### Tijd wordt niet correct aangepast (teruggaan in tijd)
#### Customer_ID achterhalen obv t_d
#### Idle servers achterhalen
#### Toewijzen aan bepaalde server

import numpy as np
import math
from collections import defaultdict
from random import random


def Exponential_distribution(lambdaValue):
    j1 = np.random.uniform(0, 1)
    if (j1 == 0): j1 += 0.0001
    j2 = -math.log(j1) / lambdaValue
    return j2


def Normal_distribution(mean, stdev):
    # TO MODEL BASED ON CUMULATIVE DENSITY FUNCTION OF NORMAL DISTRIBUTION BASED ON BOOK OF SHELDON ROSS, Simulation, The polar method, p80.

    t = 0
    while (t >= 1 or t == 0):
        r1 = np.random.uniform(0, 1) * 2 - 1  # randomNumber 1
        r2 = np.random.uniform(0, 1) * 2 - 1
        t = r1 * r1 + r2 * r2

    multiplier = math.sqrt(-2 * math.log(t) / t)
    x = r1 * multiplier * stdev + mean
    return x


def initialize_functions():  # Put all variables to zero

    ### INPUT DATA RELATED TO SYSTEM JOBS ###
    max_C = 20000
    for i1 in range(0, max_C):
        current_station[i1] = 0

    ### VARIABLES RELATED TO system SCANS ###
    for i1 in range(0, nr_stations):
        n_ws[i1] = 0  # nr of scans at a particular ws

    for i2 in range(0, K):
        mean_customers_system[i2] = 0
        tot_n[i2] = 0  # nr of scans in the system over time
        for i1 in range(0, nr_stations):
            tot_n_ws[i2][i1] = 0  # nr of scans in a ws over time

    ### PARAMETERS RELATED TO ARRIVAL OF SCANS ###
    max_AS = 5  # jobs can come from 5 different sources
    for i1 in range(0, nr_stations):
        n_a_ws[i1] = 0  # number of arrivals in a particular ws

    for i2 in range(0, K):
        mean_interarrival_time[i2] = 0
        for i6 in range(0, max_AS):
            tot_lambda[i2][i6] = 0

        for i3 in range(0, max_C):
            time_arrival[i2][i3] = 0  # time of arrival of the scan to the ancillary services
            for i1 in range(0, nr_stations):
                time_arrival_ws[i2][i1][i3] = 0  # time of arrival of a particular scan to a particular workstation
            scan_type[i3] = 0  # type of scan arriving

    for i6 in range(0, max_AS):
        t_a[i6] = 0  # time of next arrival for each source

    ### PARAMETERS RELATED TO Processing OF SCANS ###
    for i1 in range(0, nr_stations):
        n_d_ws[i1] = 0  # number of scans handled in a particular workstation
        for i6 in range(0, nr_servers[i1]):
            t_d[i1][i6] = 0  # time of next departure for each server in each workstation
            current_cust[i1][i6] = 0  # customer handled by a particular workstation and server
        # for i6 in range(0, max_C):
        # list_scan[i1][
        # i6] = -1  # list of customers processed at a particular workstation on a particular point in time

    for i2 in range(0, K):
        mean_service_time[i2] = 0  # calculated average service time
        tot_mu[i2] = 0;  # total service time generated
        for i1 in range(0, nr_stations):
            for i3 in range(0, max_C):
                time_service[i2][i1][i3] = 0  # service time per customer and per ws

    ### PARAMETERS RELATED TO waiting OF SCANS ###
    for i2 in range(0, K):
        mean_waiting_time[i2] = 0
        waiting_time[i2] = 0
        mean_customers_queue[i2] = 0
        tot_n_queue[i2] = 0
        for i1 in range(0, nr_stations):
            tot_n_queue_ws[i2][i1] = 0  # total nr of scans in queue at ws over time
            for i3 in range(0, max_C):
                waiting_time_job_ws[i2][i1][i3] = 0  # waiting time for a job on a particular ws

    ### VARIABLES RELATED TO Processed SCANS ###
    for i2 in range(0, K):
        mean_system_time[i2] = 0;
        for i3 in range(0, max_C):
            time_departure[i2][i3] = 0
            time_system[i2][i3] = 0
            for i1 in range(0, nr_stations):
                time_departure_ws[i2][i1][i3] = 0
                time_system_job_ws[i2][i1][i3] = 0

    for i3 in range(0, max_C):
        order_out[i3] = 0

    ### OTHER PARAMETERS ###
    for i2 in range(0, K):
        for i3 in range(0, nr_stations):
            for i1 in range(0, nr_servers[i3]):
                idle[i2][i3][i1] = 0
    for i3 in range(0, nr_stations):
        rho_ws[i3] = 0
        for i1 in range(0, nr_servers[i3]):
            rho_ws_s[i3][i1] = 0


def init():  # Initialisation function

    ### SET INPUT VALUES ###
    np.random.seed(0)  # ((i3+1)*K-run)
    # Ensure you each time use a different seed to get IID replications

    ### INPUT RADIOLOGY DPT ###
    global nr_stations
    nr_stations = 5  # Number of workstations

    global nr_servers
    nr_servers = {}  # Input number of servers per workstation
    nr_servers[0] = 3
    nr_servers[1] = 2
    nr_servers[2] = 4
    nr_servers[3] = 3
    nr_servers[4] = 1

    ### INPUT JOB TYPES ###
    global nr_job_types, nr_workstations_job
    nr_job_types = 4  # Number of scans, job types
    nr_workstations_job = {}  # Number of workstations per job type
    nr_workstations_job[0] = 4
    nr_workstations_job[1] = 3
    nr_workstations_job[2] = 5
    nr_workstations_job[3] = 3

    global route
    route = defaultdict(dict)  # Route to follow for each job type
    route[0][0] = 2  # JOB = 1
    route[0][1] = 0
    route[0][2] = 1
    route[0][3] = 4

    route[1][0] = 3  # JOB = 2
    route[1][1] = 0
    route[1][2] = 2

    route[2][0] = 1  # JOB = 3
    route[2][1] = 4
    route[2][2] = 0
    route[2][3] = 3
    route[2][4] = 2

    route[3][0] = 1  # JOB = 4
    route[3][1] = 3
    route[3][2] = 4

    global current_station
    current_station = {}  # Matrix that denotes the current station of a scan (sequence number)

    ### INPUT ARRIVAL PROCESS ###
    global nr_arrival_sources
    nr_arrival_sources = 2  # Number of arrival sources
    global n_a  # Number of scans arrived to the system
    n_a = 0
    global n_a_ws, t_a, t_lambda, tot_lambda, scan_type, time_arrival, time_arrival_ws, mean_interarrival_time
    t_lambda = 0
    n_a_ws = {}  # Number of scans arrived to a particular workstation
    t_a = {}  # Time of next arrival for each source
    global first_ta  # First arrival time over all sources
    first_ta = 0
    global index_arr  # Source of next arrival
    index_arr = 0
    tot_lambda = defaultdict(dict)
    scan_type = {}  # Type of scan arriving
    time_arrival = defaultdict(dict)  # Time of arrival of the scan to the ancillary services
    time_arrival_ws = defaultdict(
        lambda: defaultdict(dict))  # Time of arrival of a particular scan to a particular workstation
    mean_interarrival_time = {}

    # Arrival from radiology department
    global lamb, cum_distr_scans
    lamb = {}  # Arrival rate
    lamb[0] = 1 / 0.25  # Input arrival rate = 1/mean interarrival time
    cum_distr_scans = defaultdict(dict)  # Cumulative(!) distribution of job types per source
    cum_distr_scans[0][0] = 0.2  # SOURCE = 1
    cum_distr_scans[0][1] = 0.4
    cum_distr_scans[0][2] = 0.5
    cum_distr_scans[0][3] = 1

    # Arrival from other services
    lamb[1] = 1 / 1  # Input arrival rate = 1/mean interarrival time
    cum_distr_scans[1][0] = 0  # SOURCE = 2
    cum_distr_scans[1][1] = 0.4
    cum_distr_scans[1][2] = 0.4
    cum_distr_scans[1][3] = 1

    ### INPUT SERVICE PROCESS ###
    global perc_again
    perc_again = 0.02  # The probability that a scan should be re-evaluated at a particular workstation
    global n_d  # Number of scans handled
    global n_d_ws, t_d, mean_service_time, tot_mu, time_service, current_cust, list_scan
    n_d_ws = {}  # Number of scans handled in a particular workstation
    t_d = defaultdict(dict)  # Time of next departure for each server in each workstation
    global first_td  # First departure time over all sources
    global index_dep_station  # Station with first departure
    global index_dep_server  # Server with first departure
    mean_service_time = {}  # Calculated average service time
    global t_mu  # Generated service time
    n_d = first_td = index_dep_station = index_dep_server = t_mu = 0
    tot_mu = {}  # Total service time generated
    time_service = defaultdict(lambda: defaultdict(dict))  # Service time per customer and workstation
    current_cust = defaultdict(dict)  # Customer handles by a particular workstation and server
    list_scan = defaultdict(
        list)  # list of customers processed at a particular workstation on a particular point in time

    global mu
    mu = defaultdict(dict)  # Processing time per ws and job type
    mu[0][0] = 12  # WS1, J1
    mu[0][1] = 15
    mu[0][2] = 15
    mu[0][3] = 0
    mu[1][0] = 20

    mu[1][1] = 0  # WS2, J1
    mu[1][2] = 21
    mu[1][3] = 18
    mu[2][0] = 16

    mu[2][1] = 14  # WS3, J1
    mu[2][2] = 10
    mu[2][3] = 0

    mu[3][0] = 0  # WS4, J1
    mu[3][1] = 20
    mu[3][2] = 24
    mu[3][3] = 13

    mu[4][0] = 25  # WS5, J1
    mu[4][1] = 0
    mu[4][2] = 20
    mu[4][3] = 25

    global var
    var = defaultdict(dict)  # Processing variance per ws and job type
    var[0][0] = 2  # WS1, J1
    var[0][1] = 2
    var[0][2] = 3
    var[0][3] = 0

    var[1][0] = 4  # WS2, J1
    var[1][1] = 0
    var[1][2] = 3
    var[1][3] = 3

    var[2][0] = 4  # WS3, J1
    var[2][1] = 2
    var[2][2] = 1
    var[2][3] = 0

    var[3][0] = 0  # WS4, J1
    var[3][1] = 3
    var[3][2] = 4
    var[3][3] = 2

    var[4][0] = 5  # WS5, J1
    var[4][1] = 0
    var[4][2] = 3
    var[4][3] = 5

    global sigma
    sigma = defaultdict(dict)  # Processing stdev per ws and job type
    sigma[0][0] = np.sqrt(var[0][0])  # WS1, J1
    sigma[0][1] = np.sqrt(var[0][1])
    sigma[0][2] = np.sqrt(var[0][2])
    sigma[0][3] = np.sqrt(var[0][3])

    sigma[1][0] = np.sqrt(var[1][0])  # WS2, J1
    sigma[1][1] = np.sqrt(var[1][1])
    sigma[1][2] = np.sqrt(var[1][2])
    sigma[1][3] = np.sqrt(var[1][3])

    sigma[2][0] = np.sqrt(var[2][0])  # WS3, J1
    sigma[2][1] = np.sqrt(var[2][1])
    sigma[2][2] = np.sqrt(var[2][2])
    sigma[2][3] = np.sqrt(var[2][3])

    sigma[3][0] = np.sqrt(var[3][0])  # WS4, J1
    sigma[3][1] = np.sqrt(var[3][1])
    sigma[3][2] = np.sqrt(var[3][2])
    sigma[3][3] = np.sqrt(var[3][3])

    sigma[4][0] = np.sqrt(var[4][0])  # WS5, J1
    sigma[4][1] = np.sqrt(var[4][1])
    sigma[4][2] = np.sqrt(var[4][2])
    sigma[4][3] = np.sqrt(var[4][3])

    ### GENERAL DISCRETE EVENT SIMULATION PARAMETERS ###
    global N
    N = 1  # Number of scans (Stop criterion)
    global t  # Simulation time
    t = 1

    ### VARIABLES RELATED TO system SCANS ###
    global n  # Number of scans in the system
    n = 0
    global n_ws, mean_customers_system, tot_n, tot_n_ws
    n_ws = {}  # Number of scans at a particular workstation
    mean_customers_system = {}
    tot_n = {}  # Number of customers in the system over time
    tot_n_ws = defaultdict(dict)  # Number of customers in a workstation over time

    ### VARIABLES RELATED TO waiting OF SCANS ###
    global mean_waiting_time, waiting_time, waiting_time_job_ws, mean_customers_queue, tot_n_queue, tot_n_queue_ws
    mean_waiting_time = {}
    waiting_time = {}
    waiting_time_job_ws = defaultdict(lambda: defaultdict(dict))  # Waiting time for a job on a particular workstation

    mean_customers_queue = {}
    tot_n_queue = {}
    tot_n_queue_ws = defaultdict(dict)  # Total number of scans in queue at workstation over time

    ### VARIABLES RELATED TO Processed SCANS ###
    global time_departure, time_departure_ws, time_system, time_system_job_ws, order_out, mean_system_time
    time_departure = defaultdict(dict)
    time_departure_ws = defaultdict(lambda: defaultdict(dict))

    time_system = defaultdict(dict)
    time_system_job_ws = defaultdict(lambda: defaultdict(dict))

    order_out = {}
    mean_system_time = {}

    ### OTHER PARAMETERS ###
    global infinity, idle, rho_ws_s, rho_ws, rho
    rho = 0
    idle = defaultdict(lambda: defaultdict(dict))
    rho_ws_s = defaultdict(dict)
    rho_ws = {}

    ### VARIABLES RELATED TO CLOCK TIME ###
    global elapsed_time, time_subproblem, start_time, inter_time, project_start_time

    ### PUT ALL VARIABLES TO ZERO ###
    initialize_functions()

    ### INITIALISE SYSTEM ###

    ### DETERMINE FIRST ARRIVAL AND FIRST DEPARTURE ###
    # TO DO STUDENT    # Put all departure times for all customers to +infty
    infinity = math.inf
    for i1 in range(0, nr_stations):
        for i2 in range(0, nr_servers[i1]):
            t_d[i1][i2] = infinity

    # TO DO STUDENT    # Generate first arrival for all sources
    t_a[0] = t + Exponential_distribution(lamb[0])  # generate first arrival time from diagnostic department
    t_a[1] = t + Exponential_distribution(lamb[1])  # generate first arrival other departments
    # TO DO STUDENT    # Get next arrival
    first_ta = min(t_a[0], t_a[1])  # get first arrival out of the two sources
    if first_ta == t_a[0]:
        index_arr = 0
    else:
        index_arr = 1
    # TO DO STUDENT    # Calculate average arrival time to the system


# method to determine job type of arrival
def det_job_type(source):
    job_type = -1
    rand = random()
    if rand <= cum_distr_scans[source][0]:
        job_type = 0
    elif cum_distr_scans[source][0] < rand <= cum_distr_scans[source][1]:
        job_type = 1
    elif cum_distr_scans[source][1] < rand <= cum_distr_scans[source][2]:
        job_type = 2
    elif cum_distr_scans[source][2] < rand:
        job_type = 3
    return job_type


# method to get the ws, server and time of first departure
def get_ws_server(dict):
    res_dict = {}
    for k, v in dict.items():
        for k2, v2 in v.items():
            res_dict[(k, k2)] = v2

    min_val = min(res_dict.values())
    ws = [k for k, v in res_dict.items() if v == min_val][0][0]
    server = [k for k, v in res_dict.items() if v == min_val][0][1]
    return ws, server, min_val


# method to get next station in route for scan based on job type and current ws
def get_next_ws(current_ws, job_type):
    index_route = (list(route[job_type].values()).index(current_ws))
    return route[job_type][index_route + 1]


# method to get list of available servers at a particular ws
def get_idles(mydict):
    return [k for k, v in mydict.items() if v == 0]


def arrival_event(source):
    global n_a, n, scan_type, current_station, list_scan, n_ws, n_a_ws, t_d, t_a
    n_a += 1  # increment nr of arrivals in system
    job_type = det_job_type(source)  # determine type of job
    scan_type[n_a] = job_type  # store type of scan
    current_station[n_a] = 0  # update current ws (sequence nr)
    ws = route[job_type][0]  # help variable for current ws
    if n_ws[ws] >= nr_servers[ws]:  # check queue at first ws
        list_scan[route[job_type][current_station[n_a]]].append(n_a)  # put in queue
        # TO DO: update stat of queue
    n += 1  # update nr of scans in system currently
    n_ws[ws] += 1  # increment nr of scans at ws currently
    n_a_ws[ws] += 1  # increment nr of arrivals ws
    if n_ws[ws] < nr_servers[ws]:  # no queue
        service_time = Normal_distribution(mu[ws][job_type], var[ws][job_type])  # generate service time
        # TO DO: assign to first available server (not idle)
        t_d[ws][0] = t + service_time  # store departure time of scan at particular station and server
    # generate next arrival
    t_lamb = Exponential_distribution(lamb[source])  # generate interarrival time of next customer in system
    t_a[source] = t + t_lamb  # store time of next arrival for source


def departure_event(cust_ID):
    global n_ws, n_d_ws, n_d, current_station, t_d, n_a_ws
    job_type = scan_type[cust_ID]  # get scan type of departing cust
    current_ws = route[job_type][current_station[cust_ID]]  # get ws customer is departing from
    n_ws[current_ws] -= 1  # update nr of scans at ws
    n_d_ws[current_ws] += 1  # update nr of scans handled at particular ws
    if current_ws == route[scan_type[cust_ID]][nr_workstations_job[job_type] - 1]:  # check if final ws
        n_d += 1
    else:
        next_ws = route[job_type][current_station[cust_ID] + 1]  # identify next ws
        current_station[cust_ID] += 1  # move cust to that station
        if n_ws[next_ws] >= nr_servers[next_ws]:  # check if queue at next ws
            next_cust = list_scan[next_ws].pop(0)  # handle scan with longest wait in queue
            service_time = Normal_distribution(mu[next_ws][job_type], var[next_ws][job_type])  # generate service time
            t_d[next_ws][0] = t + service_time  # assign to first available server (not idle)
        n_ws[next_ws] += 1  # increment nr of scans at ws currently
        n_a_ws[next_ws] += 1  # increment nr of arrivals ws
        # if n_a <= N:  # stop criterion: if number of arrived customers in system < N
        if n_ws[next_ws] <= nr_servers[next_ws]:  # no queue at ws
            service_time = Normal_distribution(mu[next_ws][job_type], var[next_ws][job_type])  # generate service time
            t_d[next_ws][0] = t + t_mu  # assign to first available server (not idle)


def radiology_system():
    global t
    while n_d < N:  # perform sim until no scans are left in system
        first_departure = math.inf
        for ws, server in t_d.items():
            for time in server.values():
                if time < first_departure:
                    first_departure = time
        cust_ID = 1
        first_arrival = min(t_a[0], t_a[1])
        next_source = -1
        if first_arrival == t_a[0]:
            next_source = 0
        else:
            next_source = 1
        next_event = min(first_arrival, first_departure)
        if next_event == first_arrival:
            arrival_event(next_source)
            t = first_arrival
        else:
            departure_event(cust_ID)
            t = first_departure


# TO DO STUDENT        # Perform simulation until prespecified number of customers have departed (while loop) DONE

# TO DO STUDENT        # Identify next departure event

# TO DO STUDENT        # Identify next arrival event

# TO DO STUDENT        # Identify next event (arrival or departure)

# TO DO STUDENT        # ARRIVAL EVENT

# TO DO STUDENT        # DEPARTURE EVENT




L = 1
for i3 in range(0, L):
    K = 1
    for run in range(0, K):
        init()
        radiology_system()
        print("Nr of departures: {}".format(n_d))
        print("Time of last departure: ".format(t))
        print("Scan type of cust_ID: ".format(scan_type[1]))
        for i1 in range(0, nr_stations):
            print("Nr of departures from station {}: {}".format(i1, n_d_ws[i1]))


