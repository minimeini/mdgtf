# Source of Data

For the country-level data, we used the COVID-19 dataset from Our World in Data (https://github.com/owid/covid-19-data/tree/master/public/data), containing various metrics from different sources at the country level, including daily new cases and cases per million from the WHO's COVID Dashboard. The dataset covers 245 countries and regions from March 1st, 2020, to August 2nd, 2023, offering a comprehensive record of the pandemic's progression.

Additionally, for California county-level data, we used data from the California Department of Public Health (https://data.chhs.ca.gov/dataset/covid-19-time-series-metrics-by-county-and-state), including data on daily new cases, vaccinations, variants, and other COVID-related or demographic metrics.

To ensure compatibility with Koyama (2021) and to illustrate our models and methods effectively, we used country-level data spanning from March 1, 2020 (the month COVID was declared a pandemic by the WHO) to the end of year 2020,resulting in 276 observations in total. As for the county-level daily new cases, we used data from March 1, 2020 to March 31, 2021, totaling 396 observations. We will conduct a more comprehensive analysis using the entire dataset in the future.

Reproduction number: https://github.com/crondonm/TrackingR/tree/main/Estimates-Database


County level: 
    - : https://data.chhs.ca.gov/dataset/covid-19-time-series-metrics-by-county-and-state from California Department of Public Health, provides county-level population, and daily new cases.
    - https://covidtracking.com/analysis-updates/introducing-the-covid-tracking-project-city-dataset the COVID tracking project (CTP)
    - https://data.pa.gov/Covid-19/COVID-19-Aggregate-Cases-Current-Weekly-County-Hea/j72v-r42c Pennsylvania govermental website, provides daily new cases, daily new cases rate per 100k population, and 7-day average new case rate per 100k population.
    - https://covidactnow.org/?s=46913439 and API reference https://apidocs.covidactnow.org/data-definitions/#case-density define city/county-level case density as number of cases per 100k population calculated using a 7-day rolling average, and weekly new cases per 100k
        - API key: 75c66c903bfc45f4a882111472bc0b3e


California County to County Commute Patterns
    - https://labormarketinfo.edd.ca.gov/data/county-to-county-commute-patterns.html