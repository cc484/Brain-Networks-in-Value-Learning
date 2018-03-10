import numpy as np
from brain_state_extraction import brain_st_extr


def main():
    # generate dictionary for subject, day and learn filenames for array creation
    header = 'HOA112/dbrf/'
    subject = {}
    for s in range(20):
        s = s + 1
        subject[s] = header + 's' + str(s) + '/'

    days = {}
    for d in range(4):
        d = d + 1
        days[d] = 'day' + str(d) + '/'

    learn = {}
    for l in range(3):
        l = l + 1
        learn[l] = 'LEARN' + str(l) + '/'

    # creates condition matrix of HOA112 parcellated BOLD magnitudes
    # testing with just SUBJECT 1
    g_val = 1
    condition_matrix = np.empty((len(days), len(learn)), dtype=np.ndarray)

    for d in days:
        for l in learn:
            # converts csv to numpy array
            csv_arr = np.genfromtxt(subject[1] + days[d] + learn[l] + 'TS_HOA112.csv', delimiter=',')
            condition_matrix[d-1, l-1] = csv_arr

    # extracts brain states from all subjects during day1 brain scans
    brain_states = np.empty((len(days), len(learn)), dtype=tuple)
    for d in days:
        for l in learn:
            # stores returned tuple values from brain_state_extraction in brain_states array
            brain_states = brain_st_extr(condition_matrix, g_val)

    print('SUBJECT 1 \n')
    for d in days:
        print('day' + str(d) + ' \n')
        for l in learn:
            print(' =      ' + str(brain_states[d-1, l-1]) + ' \n')


if __name__ == "__main__":
    main()

