# Work Harder

This project came along from a necessity of my own, Irio Musskopf. Running a crowdfunding campaign on [Catarse](https://www.catarse.me/), called "[Serenata de Amor Operation](https://www.catarse.me/serenata)", I asked myself a question after a couple weeks past the publication of the draft: **should we work harder, or much harder** to be successful and raise the money we wanted?

Here are the results of my research, trying to predict the successability of a crowdfunding campaign on Catarse using Machine Learning.

## Ideas

* Collect project contributions using `src/fetch_project_contributions.py`.
* Visualize shapes of project contributions.
* Generate features based on project contributions shape (e.g. # of contributions in the first day, amount pledged in the first 2 weeks...).
* Predict using new project contributions features.
* Find clusters of projects considering their contribution stats.
* Predict outcomes for each of the project groups.
