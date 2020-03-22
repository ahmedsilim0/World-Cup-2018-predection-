import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

result=pd.read_csv('results.csv')

winner=[]
for i in range(len(result['home_team'])):
    if(result['home_score'][i]>result['away_score'][i]):
        winner.append(result['home_team'][i])
    elif(result['home_score'][i]<result['away_score'][i]):
        winner.append(result['away_team'][i])
    else:
        winner.append('Draw')

result['winner_team']=winner
result.head()
result['goal_diffrence']=np.abs(result['home_score']-result['away_score'])
result.head()

df=result[(result['home_team']=='Egypt')|(result['away_team']=='Egypt')]
egypt=df.iloc[:]
egypt.head()

year=[]
for row in egypt['date']:
    year.append(int(row[:4]))

egypt['match_year']=year
egypt_1930=egypt[egypt.match_year>=1930]
egypt_1930.count()

wins=[]

for row in egypt_1930['winner_team']:
    if row!='Egypt' and row!='Draw':
        wins.append('loss')
    else:
        wins.append(row)
winsdf=pd.DataFrame(wins,columns=['Egypt_Results'])
winsdf.head()

fig,ax=plt.subplots(1)
fig.set_size_inches(10.27, 6.27)
sns.set(style='darkgrid')
sns.countplot(x='Egypt_Results' , data=winsdf)


df2=pd.read_csv('fixtures.csv')
x=pd.DataFrame(df2)
worldcup_team=x['Home Team'][:48]
worldcup_team.unique()
print(worldcup_team)


df_team_home=result[result['home_team'].isin(worldcup_team)]
df_team_away=result[result['away_team'].isin(worldcup_team)]
df_teams=pd.concat((df_team_home, df_team_away))
df_teams.drop_duplicates()
df_teams.count()

year=[]
for row in df_teams['date']:
    year.append(int(row[:4]))

df_teams['match_year']=year
df_teams_1930=df_teams[df_teams.match_year>=1930]
df_teams_1930.head()

df_teams_1930 = df_teams.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'goal_diffrence', 'match_year'], axis=1)
df_teams_1930.head()

df_teams_1930 = df_teams_1930.reset_index(drop=True)
df_teams_1930.loc[df_teams_1930.winner_team == df_teams_1930.home_team,'winner_team']=2
df_teams_1930.loc[df_teams_1930.winner_team == 'Draw', 'winner_team']=1
df_teams_1930.loc[df_teams_1930.winner_team == df_teams_1930.away_team, 'winner_team']=0

df_teams_1930.head()

final = pd.get_dummies(df_teams_1930, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
X = final.drop(['winner_team'], axis=1)
y = final["winner_team"]
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)
print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))

ranking = pd.read_csv('fifa_rankings.csv') 
fixtures = pd.read_csv('fixtures.csv')

pred_set = []

fixtures.insert(1, 'first_position', fixtures['Home Team'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Away Team'].map(ranking.set_index('Team')['Position']))

fixtures = fixtures.iloc[:48, :]
fixtures.tail()

for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winner_team': None})
    else:
        pred_set.append({'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winner_team': None})


pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
pred_set.head()

pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]
pred_set = pred_set.drop(['winner_team'], axis=1)

pred_set.head()

predictions = logreg.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predictions[i] == 2:
        print("Winner: " + backup_pred_set.iloc[i, 1])
    elif predictions[i] == 1:
        print("Draw")
    elif predictions[i] == 0:
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
    print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1]))
    print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
    print("")