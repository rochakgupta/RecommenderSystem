__author__ = 'Shubham'
#from critics import *
import recommendations
from recommendations import critics
print critics['Lisa Rose']['Lady in the Water']
'''for it in critics['Lisa Rose']:
    print critics['Lisa Rose'][it]
    So you can see in for loop auto increment is there
    where as in while loop you have to perform the increment'''

print 'Lisa Rose:'+str(critics['Lisa Rose'])
print 'Gene Seymour:'+str(critics['Gene Seymour'])
print 'Michael Phillips:'+str(critics['Michael Phillips'])
print 'Mick LaSalle:'+str(critics['Mick LaSalle'])
print 'Claudia Puig:'+str(critics['Claudia Puig'])

print recommendations.sim_distance(recommendations.critics,'Lisa Rose','Gene Seymour')

print recommendations.sim_distance(recommendations.critics,'Lisa Rose','Michael Phillips')

print recommendations.sim_pearson(recommendations.critics,'Lisa Rose','Gene Seymour')

print recommendations.sim_pearson(recommendations.critics,'Lisa Rose','Michael Phillips')
print "Toby:"+str(critics['Toby'])

print recommendations.topMatches(recommendations.critics,'Toby',n=3)

print recommendations.getRecommendations(recommendations.critics,'Toby')

print recommendations.getRecommendations(recommendations.critics,'Toby',similarity=recommendations.sim_distance)

#print recommendations.kcluster(recommendations.critics)


reload(recommendations)
prefs=recommendations.loadMovieLens( )
print prefs['87']
print len(prefs['87'])
print recommendations.getRecommendations(prefs,'87')[0:30]
#critics=recommendations.loadDataset("")
#recommendations.sim_distance(critics,'98556', '180727')
#print recommendations.sim_distance(critics,'82444', '255618')
#print critics['228054']