import math
from collections import defaultdict
import matplotlib.pyplot as plt
import statistics
import numpy as np

# TODO: objective function per run (avg of normal and antithetic)
# TODO: running average cycle times


def Exponential_distribution(lambdaValue):
    global interarrival_times, j1_c
    j1 = np.random.uniform(0, 1)
    if (n_a % 2) == 0:
        j1_c = 1 - j1
    if (n_a % 2) == 1:
        j1 = j1_c
    if (j1 == 0): j1 += 0.0001
    j2 = -math.log(j1) / lambdaValue
    interarrival_times.append(j2)
    return j2


def Normal_distribution(mean, stdev, station, ID):
    # TO MODEL BASED ON CUMULATIVE DENSITY FUNCTION OF NORMAL DISTRIBUTION BASED ON BOOK OF SHELDON ROSS, Simulation, The polar method, p80.
    global service_times, counter, complement_random_nr1, complement_random_nr2
    t1 = 0
    if (station % 2) == 0:
        while (t1 >= 1 or t1 == 0):
            random_nr1 = np.random.uniform(0, 1)
            r1 = random_nr1 * 2 - 1  # randomNumber 1
            random_nr2 = np.random.uniform(0, 1)
            r2 = random_nr2 * 2 - 1
            t1 = r1 * r1 + r2 * r2
    if (station % 2) == 0:
        complement_random_nr1[ID] = 1 - random_nr1
        complement_random_nr2[ID] = 1 - random_nr2
    if (station % 2) == 1:
        random_nr1 = complement_random_nr1[ID]
        r1 = random_nr1 * 2 - 1  # randomNumber 1
        random_nr2 = complement_random_nr2[ID]
        r2 = random_nr2 * 2 - 1
        t1 = r1 * r1 + r2 * r2

    multiplier = math.sqrt(-2 * math.log(t1) / t1)
    x = r1 * multiplier * stdev + mean
    service_times.append(x)
    if x <= 0:
        x = 0.00001
    if (station % 2) == 0:
        service_times_normal.append(x)
    else:
        service_times_anti.append(x)
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

    for i4 in range(0, max_C):
        complement_random_nr1[i4] = 0
        complement_random_nr2[i4] = 0



def init():  # Initialisation function

    global counter, j1_c, complement_random_nr1, complement_random_nr2
    counter = 0
    j1_c = 0
    complement_random_nr1 = {}
    complement_random_nr2 = {}

    ### SET INPUT VALUES ###

    ### INPUT RADIOLOGY DPT ###
    global nr_stations
    nr_stations = 5  # Number of workstations

    global nr_servers
    nr_servers = {}  # Input number of servers per workstation
    nr_servers[0] = 3
    nr_servers[1] = 2    # experiment with nr of servers
    nr_servers[2] = 4
    nr_servers[3] = 3
    nr_servers[4] = 1   # experiment with nr of servers

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
    index_arr = {}
    tot_lambda = defaultdict(dict)
    scan_type = {}  # Type of scan arriving
    time_arrival = defaultdict(dict)  # Time of arrival of the scan to the ancillary services
    time_arrival_ws = defaultdict(
        lambda: defaultdict(dict))  # Time of arrival of a particular scan to a particular workstation
    mean_interarrival_time = {}

    # Arrival from radiology department
    global lamb, cum_distr_scans
    lamb = {}  # Arrival rate
    lamb[0] = 1 / 0.25 / 60  # Input arrival rate = 1/mean interarrival time -> converted to minutes
    cum_distr_scans = defaultdict(dict)  # Cumulative(!) distribution of job types per source
    cum_distr_scans[0][0] = 0.2  # SOURCE = 1
    cum_distr_scans[0][1] = 0.4
    cum_distr_scans[0][2] = 0.5
    cum_distr_scans[0][3] = 1

    # Arrival from other services
    lamb[1] = 1 / 1 / 60  # Input arrival rate = 1/mean interarrival time -> converted to minutes
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
    global infinity, idle, rho_ws_s, rho_ws, rho, running_avg_rho
    rho = 0
    idle = defaultdict(lambda: defaultdict(dict))
    rho_ws_s = defaultdict(dict)
    rho_ws = {}
    running_avg_rho = []


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
    rand = np.random.uniform(0, 1)
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
    return idles[0]


def arrival_event(source):
    global n_a, n, scan_type, current_station, list_scan, n_ws, n_a_ws, t_d, t_a, t, t_lambda, current_cust, rho_ws_s, index_arr
    n_a += 1  # increment nr of arrivals in system (also serves as a customer ID)
    if n_a == 341:
        test = 0
    index_arr[n_a] = source
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
        service_time = Normal_distribution(mu[ws][job_type], var[ws][job_type], station=0, ID=n_a)  # generate service time
        first_available_server = get_idles(idle[run][ws], t)  # assign to first available server (not idle)
        t_d[ws][first_available_server] = t + service_time  # store departure time of scan at particular station and
        # server
        rho_ws_s[ws][first_available_server] += t - idle[run][ws][first_available_server]  # store idle time
        idle[run][ws][first_available_server] = t + service_time  # update time server will be idle again
        current_cust[ws][first_available_server] = n_a  # update current_cust variable
    n_ws[ws] += 1  # increment nr of scans at ws currently
    # generate next arrival
    t_lambda = Exponential_distribution(lamb[source])  # generate interarrival time of next customer in system
    t_a[source] = t + t_lambda  # store time of next arrival for source


def departure_event(cust_ID):
    global n_ws, n_d_ws, n_d, current_station, t_d, n_a_ws, t, current_cust, order_out, rho_ws_s, running_avg_rho
    job_type_dep = scan_type[cust_ID]  # get scan type of departing cust
    current_ws = route[job_type_dep][current_station[cust_ID]]  # get ws customer is departing from
    n_ws[current_ws] -= 1  # update nr of scans at ws
    n_d_ws[current_ws] += 1  # update nr of scans handled at particular ws
    if n_ws[current_ws] >= nr_servers[current_ws]:  # there are people in queue at ws cust is departing from
        next_customer = list_scan[current_ws].pop(0)  # treat next scan in queue
        job_type_arr = scan_type[next_customer]  # get type of that scan
        service_time = Normal_distribution(mu[current_ws][job_type_arr],
                                           var[current_ws][job_type_arr], station=current_station[next_customer], ID=next_customer)  # generate service time
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
        # store running avg rho
        rho_ws_now = {}
        rho_now = 0
        for k1 in range(0, nr_stations):
            rho_ws_now[k1] = 0
        for k1 in range(0, nr_stations):
            for k2 in range(0, nr_servers[k1]):
                rho_ws_now[k1] += (1 - rho_ws_s[k1][k2] / t) / nr_servers[k1]
        # get rho of this run
        for k1 in range(0, nr_stations):
            rho_now += rho_ws_now[k1] / nr_stations
        if cust_ID <= N:
            running_avg_rho.append(rho_now)
    else:
        next_ws = route[job_type_dep][current_station[cust_ID] + 1]  # identify next ws (not sequence nr)
        n_a_ws[next_ws] += 1  # increment nr of arrivals ws
        current_station[cust_ID] += 1  # move cust to that station
        if n_ws[next_ws] >= nr_servers[next_ws]:  # no servers available
            list_scan[next_ws].append(cust_ID)  # add customer to that queue
        # if n_a <= N:  # stop criterion: if number of arrived customers in system < N
        if n_ws[next_ws] < nr_servers[next_ws]:  # servers available
            service_time = Normal_distribution(mu[next_ws][job_type_dep],
                                               var[next_ws][job_type_dep], station=current_station[cust_ID], ID=cust_ID)  # generate service time
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
        t = next_event  # advance time to next event
        if next_event == first_arrival:  # arrival event
            arrival_event(next_source)
        else:  # departure event
            departure_event(cust_ID)
        not_departed = []
        for i1 in range(1, N + 1):
            if order_out[i1] == 0:
                not_departed.append(i1)  # to check if first N scans have departed



L = 10
service_times = []
interarrival_times = []
rhos = []
cycle = []
service_times_normal = []
service_times_anti = []
performance_measure_batch = []
for batch in range(0, L):
    K = 1
    performance_measure = []
    for run in range(0, K):
        global rho_ws, rho, rho_ws_s
        seed = (batch + 31) * K - run
        np.random.seed(seed)  # set different random seed for every run
        init()
        radiology_system()
        # get rho of this run for all stations
        for i1 in range(0, nr_stations):
            for i2 in range(0, nr_servers[i1]):
                if rho_ws_s[i1][i2] == 0:
                    rho_ws_s[i1][i2] = t
        for i1 in range(0, nr_stations):
            for i2 in range(0, nr_servers[i1]):
                rho_ws[i1] += (1 - rho_ws_s[i1][i2] / t) / nr_servers[i1]
        # get rho of this run
        for i1 in range(0, nr_stations):
            rho += rho_ws[i1] / nr_stations
        rhos.append(rho)
        # get avg cycle time of this run
        cycle_times = []
        for i1 in range(1, N + 1):
            cycle_times.append(time_system[run][i1])
        running_avg_cycle = []
        for i1 in range(1, N + 1):
            running_avg_cycle.append(statistics.mean(cycle_times[:i1]))
        cycle_avg = statistics.mean(cycle_times)
        cycle.append(cycle_avg)
        obj_function_run = cycle_avg - 10*rho
        performance_measure.append(obj_function_run)
        # get running average of objective function for this run
        running_avg_obj = []
#        for i1 in range(0, N):
#            running_avg_obj.append(running_avg_cycle[i1] - 10*rhos[i1])
        # plot running avg of objective function
#        limit_lower = obj_function_run*0.990
#        limit_upper = obj_function_run*1.010
#        plt.plot(running_avg_obj)  # for later
#        plt.hlines(limit_upper, 1, N+1, colors='red')
#        plt.hlines(limit_lower, 1, N+1, colors='red')
#        plt.title("Run {}".format(batch))
#        plt.show()
    #avg performance measure over runs
    performance_measure_batch.append(statistics.mean(performance_measure))
performance_measure_final = statistics.mean(performance_measure_batch)
print("The performance measure of this design is: {}".format(performance_measure_final))
print("Standard deviation of system is: {}".format(statistics.stdev(performance_measure_batch)))
file1 = open("Output_Radiology.txt", "w")
file1.write("Performance measure of this design: {}\n\n\n".format(performance_measure_final))
file1.write("Performance measure per simulation:\n\n")
file1.write("Replication\t\tPerformance measure\n")
for i1 in range(0, L):
    file1.write("{}\t\t\t\t".format(i1))
    file1.write("{}\n".format(performance_measure_batch[i1]))









