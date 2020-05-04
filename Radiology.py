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

    t1 = 0
    while (t1 >= 1 or t1 == 0):
        r1 = np.random.uniform(0, 1) * 2 - 1  # randomNumber 1
        r2 = np.random.uniform(0, 1) * 2 - 1
        t1 = r1 * r1 + r2 * r2

    multiplier = math.sqrt(-2 * math.log(t1) / t1)
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
    np.random.seed(0)  # ((i3+1)*run-run)
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
    N = 1000  # Number of scans (Stop criterion)
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
    # Put all departure times for all customers to +infty
    infinity = math.inf
    for i1 in range(0, nr_stations):
        for i2 in range(0, nr_servers[i1]):
            t_d[i1][i2] = infinity

    # Generate first arrival for all sources
    t_a[0] = t + Exponential_distribution(lamb[0])  # generate first arrival time from diagnostic department
    t_a[1] = t + Exponential_distribution(lamb[1])  # generate first arrival other departments


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


# method to get the ws, server of first departure
def get_ws_server(dict, time_departure):
    global index_dep_station, index_dep_server
    for ws, server in dict.items():
        for server2, time in server.items():
            if time == time_departure:
                index_dep_server = server2
                index_dep_station = ws
    return index_dep_station, index_dep_server


# method to get first available server at a particular ws
def get_idles(mydict, current_time):
    idles = [k for k, v in mydict.items() if v <= current_time]
    test = 0
    return idles[0]


def arrival_event(source):
    global n_a, n, scan_type, current_station, list_scan, n_ws, n_a_ws, t_d, t_a, t, t_lambda, current_cust, rho_ws_s
    n_a += 1  # increment nr of arrivals in system (also serves as a customer ID)
    job_type = det_job_type(source)  # determine type of job
    scan_type[n_a] = job_type  # store type of scan
    current_station[n_a] = 0  # update current ws (sequence nr)
    ws = route[job_type][0]  # help variable for current ws (not sequence nr)
    if n_a <= N:  # stop criterion
        time_arrival[run][n_a] = t  # store arrival time
    if n_ws[ws] >= nr_servers[ws]:  # there are no servers available
        list_scan[ws].append(n_a)  # put in queue
    n += 1  # update nr of scans in system currently
    n_a_ws[ws] += 1  # increment nr of arrivals ws
    if n_ws[ws] < nr_servers[ws]:  # there are servers available
        service_time = Normal_distribution(mu[ws][job_type], var[ws][job_type])  # generate service time
        first_available_server = get_idles(idle[run][ws], t)  # assign to first available server (not idle)
        t_d[ws][first_available_server] = t + service_time  # store departure time of scan at particular station and
        # server
        rho_ws_s[ws][first_available_server] += t - idle[run][ws][first_available_server]  # store idle time
        idle[run][ws][first_available_server] = t + service_time  # update time server will be idle again
        current_cust[ws][first_available_server] = n_a  # update current_cust variable
    n_ws[ws] += 1  # increment nr of scans at ws currently
    # generate next arrival
    t_lambda = Exponential_distribution(lamb[source]) * 60  # generate interarrival time of next customer in system
    t_a[source] = t + t_lambda  # store time of next arrival for source


def departure_event(cust_ID):
    global n_ws, n_d_ws, n_d, current_station, t_d, n_a_ws, t, current_cust, order_out, rho_ws_s
    job_type_dep = scan_type[cust_ID]  # get scan type of departing cust
    current_ws = route[job_type_dep][current_station[cust_ID]]  # get ws customer is departing from
    n_ws[current_ws] -= 1  # update nr of scans at ws
    n_d_ws[current_ws] += 1  # update nr of scans handled at particular ws
    if n_ws[current_ws] >= nr_servers[current_ws]:  # there are people in queue at ws cust is departing from
        next_customer = list_scan[current_ws].pop(0)  # treat next scan in queue
        job_type_arr = scan_type[next_customer]  # get type of that scan
        service_time = Normal_distribution(mu[current_ws][job_type_arr],
                                           var[current_ws][job_type_arr])  # generate service time
        first_available_server = get_idles(idle[run][current_ws],
                                           t)  # assign to first available server (not idle)
        t_d[current_ws][first_available_server] = t + service_time  # store departure time
        idle[run][current_ws][first_available_server] = t + service_time  # update time server will be idle again
        current_cust[current_ws][first_available_server] = next_customer  # update current_cust variable
    if current_ws == route[scan_type[cust_ID]][nr_workstations_job[job_type_dep] - 1]:  # check if final ws
        n_d += 1  # increment nr of departures out of system
        order_out[cust_ID] = t  # store time cust is out of system
        time_system[run][cust_ID] = order_out[cust_ID] - time_arrival[run][cust_ID]  # store time scan has spent
        # in the system
    else:
        next_ws = route[job_type_dep][current_station[cust_ID] + 1]  # identify next ws (not sequence nr)
        n_a_ws[next_ws] += 1  # increment nr of arrivals ws
        current_station[cust_ID] += 1  # move cust to that station
        if n_ws[next_ws] >= nr_servers[next_ws]:  # no servers available
            list_scan[next_ws].append(cust_ID)  # add customer to that queue
        # if n_a <= N:  # stop criterion: if number of arrived customers in system < N
        if n_ws[next_ws] < nr_servers[next_ws]:  # servers available
            service_time = Normal_distribution(mu[next_ws][job_type_dep],
                                               var[next_ws][job_type_dep])  # generate service time
            first_available_server = get_idles(idle[run][next_ws],
                                               t)  # assign to first available server (not idle)
            rho_ws_s[next_ws][first_available_server] += t - idle[run][next_ws][first_available_server]  # store
            # idle time
            t_d[next_ws][first_available_server] = t + service_time  # store departure time
            idle[run][next_ws][first_available_server] = t + service_time  # update time server will be idle again
            current_cust[next_ws][first_available_server] = cust_ID  # update current_cust variable
        n_ws[next_ws] += 1  # increment nr of scans at ws currently


def radiology_system():
    global t
    not_departed = [1]
    while len(not_departed) != 0:  # perform sim until first N scans have departed
        first_departure = math.inf  # initialize first departure variable
        for ws, server in t_d.items():  # get time of first next departure
            for time in server.values():
                if t < time < first_departure:
                    first_departure = time
        index_dep_station, index_dep_server = get_ws_server(t_d, first_departure)  # get station and server next
        # departure is departing from
        cust_ID = current_cust[index_dep_station][index_dep_server]  # get customer ID of next departing customer
        first_arrival = min(t_a[0], t_a[1])
        if first_arrival == t_a[0]:  # get source of next arrival
            next_source = 0
        else:
            next_source = 1
        next_event = min(first_arrival, first_departure)
        t = next_event
        if next_event == first_arrival:  # arrival event
            arrival_event(next_source)
        else:  # departure event
            departure_event(cust_ID)
        not_departed = []
        for i1 in range(1, N + 1):
            if order_out[i1] == 0:
                not_departed.append(i1)  # to check if first N scans have departed


def output():
    global rho_ws, rho, t, mean_system_time
    file1 = open("Output_Radiology_run{}.txt".format(run), "w")
    for i1 in range(1, N + 1):  # PRINT system time = cycle time (observations and running average)
        mean_system_time[run] += time_system[run][i1]
    file1.write('Cycle time run {}:\n\n'.format(run))
    j1 = mean_system_time[run] / N
    file1.write('Avg cycle time: %lf\n\n\n' % j1)
    mean_system_time[run] = 0
    file1.write('Number\tObservation\tRunning Average\tScan type\n')

    for i1 in range(1, N + 1):
        mean_system_time[run] += time_system[run][i1]
        j1 = mean_system_time[run] / (i1 + 1)
        file1.write('%d\t' % i1 + '%lf\t' % time_system[run][i1] + '%lf\t' % j1 + '%d\n' % scan_type[i1])
    file1.write('\n\n\n')

    file1.write('Utilization level run {}:\n\n'.format(run))  # Print stat utilization lvl's
    for i1 in range(0, nr_stations):
        file1.write('Utilisation servers Station WS %d:\t' % i1)
        for i2 in range(0, nr_servers[i1]):
            file1.write('%lf\t' % (1 - rho_ws_s[i1][i2] / t))
        file1.write('\n')
    file1.write('\n')
    for i1 in range(0, nr_stations):
        file1.write('Avg utilisation Station WS %d:\t' % i1)
        for i2 in range(0, nr_servers[i1]):
            rho_ws[i1] += (1 - rho_ws_s[i1][i2] / t)
        rho_ws[i1] = rho_ws[i1] / nr_servers[i1]
        file1.write('%lf\n' % rho_ws[i1])
    file1.write('\n')

    for i1 in range(0, nr_stations):
        rho += rho_ws[i1]
    rho /= nr_stations
    file1.write('Overall avg utilisation: %lf\n' % rho)
    file1.write('\n\n\n')

    function_objective = mean_system_time[run]/N/60 - 10 * rho

    file1.write('Objective function of run {}: '.format(function_objective))

L = 1
for i3 in range(0, L):
    K = 5
    for run in range(0, K):
        init()
        radiology_system()
        output()
