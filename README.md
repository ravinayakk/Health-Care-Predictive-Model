[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uFG8FNyr)
# Final-Project
<!-- Edit the title above with your project title -->

## Project Overview

    * Setup and Layout
      - When you open the repository go to the source.ipynb and run the first cell to import all the libraries
      - Then accordingly run each cell and see the visualization and for reference refer to the Readme for insights on those                    viusalizations
      - The information and details about each graph and visualization can be found on the ReadMe file(checkpoint #2)
      - The storage of the data files are located in the assets/Data folder
      
## Self Assessment and Reflection
<!-- Edit the following section with your reflection -->

   - Overall the experience and progess during the semester while working on this project has been very insightful. Throughout the progression of this project I have learned about multiple skills and concepts about machine learning and data analytics.

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | ... / 10 |
| **Execution**     | ... / 20 |
| **Documentation** | ... / 10 |
| **Presentation**  | ... / 30 |
| **Total**         | ... / 70 |

### Reflection
<!-- Edit the following section with your reflection -->

#### What went well?
   - Being able to find some data sources that could be used to display the data I needed without much cleaning
#### What did not go well?
   - Most of the data sources were troublesome in being able to extract the proper data from them they had data columns and points that      were not needed
#### What did you learn?
   - I learned many machine learning and data extraction skills as well as using them to create visualization and gain insights from         them
#### What would you do differently next time?
   - Next time I would spend more time on finding proper data sources and possibly narrow in on a more specific topic because this topic was still broad and the data surrounding it was difficult to acquire and use. 

---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Checkpoint 1: Project Idea and datasets

* Topic: What problem are you (or your stakeholder) trying to address? Why is that topic important?
  - The problem(s) I am trying to address are healthcare disparities that exist due to factors like geography, income level, insurance coverage, and availability of healthcare professionals. Communities like rural or low-income urban areas, face  challenges in accessing quality healthcare services, leading to poor health outcomes and increased health disparities. 
  - This topic is important because it address the problems within the healthcare world and focuses on low-income individuals, minorities, and rural residents, who often experience the worst health outcomes. Understanding these disparities can inform better policy decisions, funding allocations, and healthcare interventions. 

* Project Questions: What questions are you trying to answer with this project?
   - How do income, geography, insurance status, and demographics affect healthcare availability? Are rural areas more affected than urban areas? 
   - What is the correlation between healthcare access and disease prevalence, life expectancy, or infant mortality? 
   - How does access affect the rate of preventable diseases and hospitalizations?
   - What geographic areas are most in need of healthcare access improvements?


* What would an answer look like? 
  - <img src="./Images/IMG_0771.JPG">

* Data Sources:
  - What data sources have you identified for your project? You need to identify 3 data sources of at least 2 data source types (files, databases, API, scrapped pages) You also need to indicate how you think you can relate these datasets
  - I have identified 3 data sources of at least 2 data sources, the first one is the CDC Wonder API(https://wonder.cdc.gov/), the next one is the HRSA Database(https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://data.hrsa.gov/topics/health-workforce/shortage-areas&ved=2ahUKEwiY0a2N3siLAxXD78kDHejeJ04QFnoECBYQAQ&usg=AOvVaw1zR_2IQdnKu6jtD0tct4Jg), and the last one I found was a csv file U.S. Census Bureau Health Insurance Coverage Dataset(https://www.census.gov/data/data-tools/sahie-interactive.html)
  - The data sets are quite large so before I insert them into the project I will narrow them down and simplify them. 
  - By joining these datasets, I can map underserved regions and identify trends in health disparities, providing insights for policy interventions and healthcare planning.


## Checkpoint 2: Exploratory Data Analysis & Visualization 

* Exploratory Data Analysis (EDA)
  - What insights and interesting information are you able to extract at this stage?
    - Regions with lower median income tend to have lower insurance coverage, which may contribute to reduced access to healthcare services. Rural areas have significantly fewer healthcare facilities per capita compared to urban areas, highlighting a clear disparity in healthcare access.
  - Are there any correlations between my variables?
    - There is a strong positive correlation between median income and insurance coverage, meaning that regions with higher income levels tend to have higher insurance coverage. Insurance coverage and healthcare access are also positively correlated with life expectancy and negatively correlated with infant mortality rates.
  - What are the distributions of my variables?
    - The distribution of insurance coverage is slightly right-skewed, with most regions having coverage rates between 70% and 90%. However, some regions have very low coverage rates (below 50%). Mortality rates are relatively normally distributed, with most regions having rates between 5 and 10 deaths per 1,000 people. However, some regions have unusually high mortality rates, which may indicate severe healthcare disparities.

* Data Visualizations 
  - Histogram
    - Description: This histogram shows the distribution of insurance coverage rates and median income levels across regions.
    - Insights: Identify regions with low insurance coverage and low median income, which are likely to face significant healthcare access challenges. Observe whether there is a correlation between income levels and insurance coverage (lower-income regions may have lower insurance coverage).
  - Heatmap
    - Description: This heatmap visualizes the correlation coefficients between variables such as healthcare access, insurance coverage, income levels, and health outcomes (mortality rates, disease prevalence).
    - Insights: Identify strong positive or negative correlations (higher insurance coverage may correlate with lower mortality rates). Highlight which factors ( income, healthcare facilities per capita) have the strongest impact on health outcomes.
  - Bar Chart
    - Description: This bar chart compares the availability of healthcare facilities (per capita) in rural and urban areas.
    - Insights: Highlight disparities in healthcare access between rural and urban regions. Provide evidence for the need to allocate more resources to rural areas.
  -  Treemap
     - Description: This  Treemap visualizes the geographic distribution of healthcare shortages across U.S. states, using a "shortage score" to indicate severity.
     - Insights: Identify states or regions with the most severe healthcare shortages. Provide a spatial perspective on disparities, which can inform targeted interventions. The treemap will give         you a clear view of which counties are most affected by healthcare provider shortages. Larger blocks indicate counties that have significant shortages, and smaller blocks indicate those             with lesser shortages

* Data Cleaning and Transformation
  - The process I went through to prepare my data for analysis was checking for missing values and deciding how to handle them, drop or fill. 

  ## Checkpoint 3: Machine Learning (Regression/Classification)

  * Machine Learning Plan
    - The types of machine learning models I plan to use are predictive analysis model. This model would allow me to see the rate of admissions at certain hospitals in certain areas which income is low versus high. 
    - One challlenge I am anticipating in building my machine learning model is algorithmic bias. Where because it was primarily trained on admissions of patients in one certain area where the income is generally lower. But if I use it on a different group it might perform poorly on other areas/groups. 
    - I am planning to address these challenges by training the model appropriately and making sure that the model has equal training and testing data for each group. 

    * Machine Learning Implementation Process
      -  This machine learning implementation implements a model over healthcare disparities. Beginning with exploratory data analysis to identify patterns and data quality issues across CDC mortality, HRSA shortage, and Census insurance datasets. 
       - The modeling approach uses algorithms like Linear Regression. I also did train-test splitting and cross-validation techniques that account for geographic dependencies. The final model selection is based on  performance metrics (RMSE). 

  ## Project Presentation Conclusion
  
    * Challenges & Learnings
      - Challenges/Learnings: Public datasets like HRSA, CDC Wonder, and the U.S. Census can be hard to locate, download, or access via         APIs
        - HRSA’s datasets often have scattered files or limited documentation, making it hard to know which one to use
      - My project corrupted losing all my code and graphs and data. Had to restart and recollect all my data that I originally filtered
      - Libraries like seaborn, matplotlib, plotly, or scipy often throw vague errors
      - Linking datasets by county or geographic unit requires careful matching


## Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)
* https://catalog.data.gov/dataset/health-professional-shortage-areas-in-california-9cd7d
* https://data.hrsa.gov/tools/shortage-area/hpsa-find
* https://odh.ohio.gov/explore-data-and-stats
* https://bhw.hrsa.gov/data-research/projecting-health-workforce-supply-demand
