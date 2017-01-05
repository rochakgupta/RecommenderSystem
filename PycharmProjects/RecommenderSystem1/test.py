__author__ = 'rochak'
import recommendations
from recommendations import critics
import numpy, matplotlib, scipy

# print critics['Lisa Rose']['Lady in the Water']
# print ''

# critics['Toby']['Snakes on a Plane']=4.5
# print critics['Lisa Rose']
# print ''

# print critics['Lisa Rose']
#
# print critics['Gene Seymour']
#
# print critics['Michael Phillips']

# print recommendations.sim_distance(critics,'Lisa Rose','Gene Seymour')
#
# print recommendations.sim_distance(critics,'Lisa Rose','Michael Phillips')
#
# print recommendations.sim_pearson(critics,'Lisa Rose','Gene Seymour')
#
# print recommendations.sim_pearson(critics,'Lisa Rose','Michael Phillips')
#
# print recommendations.sim_cosine(critics,'Lisa Rose','Gene Seymour')
#
# print recommendations.sim_cosine(critics,'Lisa Rose','Michael Phillips')
#
# print recommendations.sim_exJaccard(critics,'Lisa Rose','Gene Seymour')
#
# print recommendations.sim_exJaccard(critics,'Lisa Rose','Michael Phillips')
#
# print recommendations.sim_Jaccard(critics,'Lisa Rose','Gene Seymour')
#
# print recommendations.sim_Jaccard(critics,'Lisa Rose','Michael Phillips')
#
# print recommendations.sim_asycos(critics,'Lisa Rose','Michael Phillips')
#
# print recommendations.sim_asycos(critics,'Michael Phillips','Lisa Rose')
#
# print recommendations.sim_asymsd(critics,'Lisa Rose','Michael Phillips')
#
# print recommendations.sim_asymsd(critics,'Michael Phillips','Lisa Rose')

# print ''

# print recommendations.topMatches(critics,'Toby', 4, similarity=recommendations.sim_distance)
# print ''

# print recommendations.getRecommendations(critics,'Toby')
# print ''

# print recommendations.getRecommendations(critics,'Toby',similarity=recommendations.sim_distance)
# print ''

# movies = recommendations.transformPrefs(critics)
# print movies
# print ''
# print recommendations.topMatches(movies,'Superman Returns')
# print ''
# print recommendations.getRecommendations(movies,'Just My Luck')
# print ''

# itemsim =  recommendations.calculateSimilarItems(critics)
# print itemsim
# print ''
# print recommendations.getRecommendedItems(critics,itemsim,'Toby')
# print ''

# def loadMovieLens(path='E:\PycharmProjects\RecommenderSystem\ml-100k'):
# for line in open(path+'\u.item'):
# (id,title,date,nothing,url)=line.split('|')[0:5]
# print str(id+':'+title+':'+date+':'+url)
#
# loadMovieLens()
# print ''

# prefs = recommendations.loadMovieLens()

# # user based recommendations
# print recommendations.getRecommendations(prefs,'87')[0:20]
# print ''

# item based recommendations
# itemsim = recommendations.calculateSimilarItems(prefs, n=40)
# print recommendations.getRecommendedItems(prefs, itemsim, '87')[0:20]
# print ''


# print recommendations.critics
# print ''
# n = recommendations.normalize(recommendations.critics)
# print n
# print ''
# print recommendations.deNormalize(n)

# my = {"Amy":
#           {"Taylor Swift": 4,
#            "PSY": 3,
#            "Whitney Houston": 4},
#       "Ben":
#           {"Taylor Swift": 5,
#            "PSY": 2},
#       "Clara":
#           {"PSY": 3.5,
#            "Whitney Houston": 4},
#       "Daisy":numpy.savetxt('training.txt', ch, delimiter=',')
#           {"Taylor Swift": 5,
#            "Whitney Houston": 3}
# }
#
# print recommendations.slopeOne(my, 'Daisy', 'PSY')

a = numpy.zeros((3, 3))
a[0][0] = 1
a[0][1] = 2
a[0][2] = 3
a = a[0, :]
a = numpy.where(a > 0)[0]
print a