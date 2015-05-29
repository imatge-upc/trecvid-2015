import numpy as np

def relnotrel( fileGT, id_q, rankingShots ):

    '''Takes ground truth file (fileGT), query name (id_q) and ranking (rankingShots) in order to create a vector of 1's and 0's to compute Average Precision
        Returns: a list of 1 and 0 for the rankingShots and the number of relevant samples in the ground truth file for the given query.
    '''
    a = np.loadtxt( fileGT, dtype='string' )

    # Extract shots for the query
    t_shot = a[ (a[:,0]==id_q) ]

    # Extract relevant shots for the query
    t_shot_rel = t_shot[ t_shot[:,3] == '1' ]
    t_shot_notrel = t_shot[ t_shot[:,3] == '0' ]


    # Total Number of relevant shots in the ground truth
    nRelTot = np.shape( t_shot_rel )[0]


    labelRankingShot = np.zeros((1, len(rankingShots)))

    i = 0
    for shotRanking in rankingShots:

        if shotRanking in t_shot_rel:
            labelRankingShot[0, i ] = 1

        i +=1


    return labelRankingShot, nRelTot


def AveragePrecision( relist,nRelTot):
    '''Takes a list of 1 and 0 and the number of relevant samples and computes the average precision'''

    accu = 0
    numRel = 0

    for k in range(len(relist)):

        if relist[k] == 1:
            numRel = numRel + 1

            accu += float( numRel )/ float(k+1)

    return (accu/nRelTot)