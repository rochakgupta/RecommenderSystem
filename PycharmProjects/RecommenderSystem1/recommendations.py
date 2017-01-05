__author__ = 'rochak'

movies = {'Marcel Caraciolo':
              {'Lady in the Water': 2.5,
               'Snakes on a Plane': 3.5,
               'Just My Luck': 3.0,
               'Superman Returns': 3.5,
               'You, Me and Dupree': 2.5,
               'The Night Listener': 3.0},
          'Luciana Nunes':
              {'Lady in the Water': 3.0,
               'Snakes on a Plane': 3.5,
               'Just My Luck': 1.5,
               'Superman Returns': 5.0,
               'The Night Listener': 3.0,
               'You, Me and Dupree': 3.5},
          'Leopoldo Pires':
              {'Lady in the Water': 2.5,
               'Snakes on a Plane': 3.0,
               'Superman Returns': 3.5,
               'The Night Listener': 4.0},
          'Lorena Abreu':
              {'Snakes on a Plane': 3.5,
               'Just My Luck': 3.0,
               'The Night Listener': 4.5,
               'Superman Returns': 4.0,
               'You, Me and Dupree': 2.5},
          'Steve Gates':
              {'Lady in the Water': 3.0,
               'Snakes on a Plane': 4.0,
               'Just My Luck': 2.0,
               'Superman Returns': 3.0,
               'The Night Listener': 3.0,
               'You, Me and Dupree': 2.0},
          'Sheldom':
              {'Lady in the Water': 3.0,
               'Snakes on a Plane': 4.0,
               'The Night Listener': 3.0,
               'Superman Returns': 5.0,
               'You, Me and Dupree': 3.5},
          'Penny Frewman':
              {'Snakes on a Plane': 4.5,
               'You, Me and Dupree': 1.0,
               'Superman Returns': 4.0}
}

critics = {'Lisa Rose':
               {'Lady in the Water': 2.5,
                'Snakes on a Plane': 3.5,
                'Just My Luck': 3.0,
                'Superman Returns': 3.5,
                'You, Me and Dupree': 2.5,
                'The Night Listener': 3.0},
           'Gene Seymour':
               {'Lady in the Water': 3.0,
                'Snakes on a Plane': 3.5,
                'Just My Luck': 1.5,
                'Superman Returns': 5.0,
                'The Night Listener': 3.0,
                'You, Me and Dupree': 3.5},
           'Michael Phillips':
               {'Lady in the Water': 2.5,
                'Snakes on a Plane': 3.0,
                'Superman Returns': 3.5,
                'The Night Listener': 4.0},
           'Claudia Puig':
               {'Snakes on a Plane': 3.5,
                'Just My Luck': 3.0,
                'The Night Listener': 4.5,
                'Superman Returns': 4.0,
                'You, Me and Dupree': 2.5},
           'Mick LaSalle':
               {'Lady in the Water': 3.0,
                'Snakes on a Plane': 4.0,
                'Just My Luck': 2.0,
                'Superman Returns': 3.0,
                'The Night Listener': 3.0,
                'You, Me and Dupree': 2.0},
           'Jack Matthews':
               {'Lady in the Water': 3.0,
                'Snakes on a Plane': 4.0,
                'The Night Listener': 3.0,
                'Superman Returns': 5.0,
                'You, Me and Dupree': 3.5},
           'Toby':
               {'Snakes on a Plane': 4.5,
                'You, Me and Dupree': 1.0,
                'Superman Returns': 4.0}
}

import numpy
from math import sqrt

"""
first step : find the cosine similarity between every user-user pair
        (subtracting from every value being considered, the mean rating for each user)
second step : predict ratings for a given user after converting each rating to its normalized (-1 to 1) version
        (the resultant normalized value obtained is again denormalized to bring back to 1 to 5 scale)
The next 3 functions are used for this only
"""


def normalize(prefs):
    minr = 1
    maxr = 5
    for user in prefs:
        for movie in prefs[user]:
            prefs[user][movie] = 2 * (prefs[user][movie] - minr) / float(maxr - minr) - 1.0

    return prefs


def deNormalize(prefs):
    minr = 1
    maxr = 5
    for user in prefs:
        for movie in prefs[user]:
            prefs[user][movie] = 0.5 * (prefs[user][movie] + 1.0) * (maxr - minr) + minr

    return prefs


def getMean(prefs):
    avg = {}
    for user in prefs:
        avg[user] = 0.0
        for movie in prefs[user]:
            avg[user] += prefs[user][movie]

        avg[user] /= len(prefs[user])

    return avg


"""
The next 2 functions are for weighted slope one prediction method
first step : find deviations and frequencies for each item-item pair
second step : predict the rating that u user would give for item i
"""


def computeDeviations(prefs):
    frequencies = {}
    deviations = {}
    for ratings in prefs.values():
        for (item1, rating1) in ratings.items():
            frequencies.setdefault(item1, {})
            deviations.setdefault(item1, {})
            for (item2, rating2) in ratings.items():
                if item1 != item2:
                    frequencies[item1].setdefault(item2, 0)
                    deviations[item1].setdefault(item2, 0.0)
                    frequencies[item1][item2] += 1
                    deviations[item1][item2] += rating1 - rating2

    for (item1, vals) in frequencies.items():
        for item2 in vals.keys():
            if frequencies[item1][item2] != 0 and deviations[item1][item2] != 0:
                deviations[item1][item2] /= frequencies[item1][item2]

    return frequencies, deviations


def slopeOne(prefs, u, i):
    frequencies, deviations = computeDeviations(prefs)
    num = 0.0
    den = 0.0
    for (item, rating) in prefs[u].items():
        num += (prefs[u][item] + deviations[i][item]) * frequencies[i][item]
        den += frequencies[i][item]
    if den == 0.0:
        print "Can't predict"
        return -1
    return num / float(den)


"""
Now begins our main code
"""

# Returns a Euclidean similarity score for person1 and person2
def sim_distance(prefs, person1, person2):
    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Add up the squares of all the differences
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2)
                          for item in prefs[person1] if item in prefs[person2]])

    return 1 / (1 + sum_of_squares)


# Returns the Minkowski correlation coefficient for p1 and p2
def sim_minkowski(prefs, p1, p2, k):
    # Get the list of mutually rated items
    # Get the list of shared_items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Add up the kth powers of all the differences
    sum_of_powers = sum([pow(prefs[p1][item] - prefs[p2][item], k)
                         for item in prefs[p1] if item in prefs[p2]])

    return 1 / (1 + sum_of_powers)


# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs, p1, p2):
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # Find the number of elements
    n = len(si)
    # if they are no ratings in common, return 0
    if n == 0:
        return 0

    # Add up all the preferences
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    # Sum up the squares
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    # Sum up the products
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # Calculate Pearson score
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))

    if den == 0:
        return 0
    return num / float(den)


# Returns the Cosine similarity coefficient for p1 and p2
def sim_cosine(prefs, p1, p2):
    a = set(prefs[p1])
    b = set(prefs[p2])
    i = a & b
    p1_p2 = 0
    p1_mag = 0
    p2_mag = 0
    for item in i:
        p1_p2 += prefs[p1][item] * prefs[p2][item]
    for item in prefs[p1]:
        p1_mag += prefs[p1][item] ** 2
    for item in prefs[p2]:
        p2_mag += prefs[p2][item] ** 2

    if p1_mag == 0 or p2_mag == 0:
        return 0
    else:
        return p1_p2 / float(sqrt(p1_mag) * sqrt(p2_mag))


# Returns the Extended Jaccard  similarity coefficient between p1 and p2
def sim_exJaccard(prefs, p1, p2):
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # Find the number of elements
    n = len(si)
    # if they are no ratings in common, return 0
    if n == 0:
        return 0

    # Find prefs[p1][item]*prefs[p2][item] for each item satisfying si[item]=1
    p1_mag = 0
    p2_mag = 0
    p1_p2 = 0
    for item in si:
        if si[item] == 1:
            p1_mag += pow(prefs[p1][item], 2)
            p2_mag += pow(prefs[p2][item], 2)
            p1_p2 += prefs[p1][item] * prefs[p2][item]

    if p1_mag + p2_mag - p1_p2 == 0:
        return 0

    r = p1_p2 / (p1_mag + p2_mag - p1_p2)
    return r


# Returns the Jaccard  similarity coefficient between p1 and p2
def sim_Jaccard(prefs, p1, p2):
    a = set(prefs[p1])
    b = set(prefs[p2])
    i = a & b
    u = a | b
    if len(i) == 0 or len(u) == 0:
        return 0

    return len(i) / float(len(u))


# Returns asymmetric similarity coefficient (as mentioned in paper)
def asymmetric(prefs, p1, p2):
    a = set(prefs[p1])
    b = set(prefs[p2])
    i = a & b
    num = 2 * len(i) * len(i)
    den = len(a) * (len(a) + len(b))
    if num == 0 or den == 0:
        return 0

    return num / float(den)


# Returns asymmetric cosine similarity coefficient
def sim_asycos(prefs, p1, p2):
    return asymmetric(prefs, p1, p2) * sim_cosine(prefs, p1, p2)


# Returns asymmetric msd similarity coefficient
def sim_asymsd(prefs, p1, p2):
    a = set(prefs[p1])
    b = set(prefs[p2])
    i = a & b
    sq_sum = 0
    for item in i:
        sq_sum += (prefs[p1][item] - prefs[p2][item]) ** 2
    l = len(i)
    msd = sq_sum / float(l)
    return ((16 - msd) * asymmetric(prefs, p1, p2)) / float(16)


# Returns the best matches for person from the prefs dictionary
# Number of the results and similarity function are optional params.
def topMatches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]

    # Sort the list so the highest scores appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n]


# Gets recommendations for a person by using a weighted average
# of every other user's rankings

def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}

    for other in prefs:
        # don't compare me to myself
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in prefs[other]:
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # Create the normalized list
    rankings = [(total / simSums[item], item) for item, total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


# Function to transform Person, item - > Item, person
def transformPrefs(prefs):
    result = {}

    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # Flip item and person
            result[item][person] = prefs[person][item]

    return result


def calculateSimilarItems(prefs, n=10):
    # Create a dictionary of items showing which other items they
    # are most similar to.
    result = {}
    # Invert the preference matrix to be item-centric
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        # Status updates for large datasets
        # c+=1
        # if c%100==0:
        # print "%d / %d" % (c,len(itemPrefs))
        # Find the most similar items to this one
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    return result


def getRecommendedItems(prefs, itemMatch, user):
    userRatings = prefs[user]
    totals = {}
    simSums = {}

    # Loop over items rated by this user
    for (item, rating) in userRatings.items():

        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:
            # Ignore if this user has already rated this item
            if item2 in userRatings:
                continue
            # Weighted sum of rating times similarity
            totals.setdefault(item2, 0)
            totals[item2] += similarity * rating
            # Sum of all the similarities
            simSums.setdefault(item2, 0)
            simSums[item2] += similarity

    # Divide each total score by total weighting to get an average
    rankings = [(total / simSums[item], item) for item, total in totals.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


def loadMovieLens(path='D:\PycharmProjects\RecommenderSystem\ml-100k'):
    # Get movie titles
    movies = {}
    for line in open(path + '\u.item'):
        (id, title) = line.split('|')[0:2]
        movies[id] = title
    # Load data
    prefs = {}
    for line in open(path + '\u.data'):
        (userid, movieid, rating, ts) = line.split('\t')
        prefs.setdefault(userid, {})
        prefs[userid][movies[movieid]] = float(rating)
    return prefs


def loadMovieLensCofi(path='C:\Recommender System\PycharmProjects\RecommenderSystem1\ml-100k'):
    # Get movie titles
    movmap = {}
    for line in open(path + '\u.item'):
        (id, title) = line.split('|')[0:2]
        movmap[int(id) - 1] = title
    # Load data
    ratmat = numpy.zeros((1682, 943))
    boolmat = numpy.zeros((1682, 943))
    for line in open(path + '\u.data'):
        (userid, movieid, rating, ts) = line.split('\t')
        ratmat[int(movieid) - 1][int(userid) - 1] = rating
        boolmat[int(movieid) - 1][int(userid) - 1] = 1

    numpy.savetxt("rating matrix.csv",ratmat,delimiter=',')

    return movmap, ratmat, boolmat
