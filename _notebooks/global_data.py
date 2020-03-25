import numpy as np

from io import StringIO
import csv
import pandas as pd
from datetime import datetime
import requests

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'

def parse_data(fname):
    csv_content = requests.get(BASE_URL + fname).content
    docs = list(csv.DictReader(StringIO(csv_content.decode('utf8'))))
        
    new_docs = []
    for doc in docs:
        meta = {k: doc[k] for k in ['Province/State', 'Country/Region', 'Lat', 'Long']}
        for k, v in doc.items():
            if k in meta: continue
            new_doc = meta.copy()
            new_doc['date'] = datetime.strptime(k, '%m/%d/%y')
            if not v: continue
            new_doc['cnt'] = int(v)
            new_docs.append(new_doc)


    return (
        pd.DataFrame(new_docs)
          .rename(
              columns={
                  'Province/State': 'province',
                  'Country/Region': 'country',
                  'Lat': 'lat', 'Long': 'long'
              }
          ).groupby(['date', 'country'])
          .cnt.sum()
          .reset_index()
    )


def get_global_covid_df():
    confirmed_df = parse_data('csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
    confirmed_df = confirmed_df.rename(columns=dict(cnt='confirmed'))

    recovered_df = parse_data('csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
    recovered_df = recovered_df.rename(columns=dict(cnt='recovered'))

    death_df = parse_data('csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
    death_df = death_df.rename(columns=dict(cnt='death'))

    # merge everything together
    df = (
        confirmed_df.merge(recovered_df, on=('country', 'date'), how='left')
                    .merge(death_df, on=('country', 'date'), how='left')
                    .fillna(0)
    )
    
    # compute growth rate by country
    dfs = []
    for c in df.country.unique():
        cdf = df[df.country == c].set_index('date').sort_index()
        cdf['confirmed_growth_rate'] = np.minimum(1.6, cdf.confirmed / cdf.confirmed.shift(1)).rolling(7).mean().fillna(1.6)
        cdf['death_growth_rate'] = np.minimum(1.6, cdf.death / cdf.death.shift(1)).rolling(7).mean().fillna(1.6)
        dfs.append(cdf)
    
    df = pd.concat(dfs).reset_index()
    
    # compute days_from_first_infection and days_from_first_death
    df = df.merge(
        df[df.confirmed > 0]
        .groupby('country')
        .date.min()
        .reset_index()
        .rename(columns=dict(date='first_infaction_date'))
    )

    df['days_from_first_infection'] = (
        (df['date'] - df['first_infaction_date']).apply(lambda x: x.days)
    )

    df = df.merge(
        df[df.death > 0]
          .groupby('country')
          .date.min()
          .reset_index()
          .rename(columns=dict(date='first_death_date'))
    )

    df['days_from_first_death'] = (
        (df['date'] - df['first_death_date']).apply(lambda x: x.days)
    )
        
    return df 

