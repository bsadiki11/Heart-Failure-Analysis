library(tidyverse)
library("ggplot2")
library(gamlr)
library(dplyr)

#PART 1

#Read the Excel File.
heartfailure <- read.csv("heartfailure.csv", strings = T)

#ReName the Columns
colnames(heartfailure) <- c("Age", "Anaemia", "CPK", "Diabetes", "Ejection Fraction", "High Blood Pressure", "Platelets", "Creatinine", "Sodium", "Sex", "Smoking", "Time", "Death")

#Everything is in Correct Format: Numeric, Factor, etc.
heartfailure$Anaemia <- as.factor(heartfailure$Anaemia)
heartfailure$Diabetes <- as.factor(heartfailure$Diabetes)
heartfailure$`High Blood Pressure` <- as.factor(heartfailure$`High Blood Pressure`)
heartfailure$Sex <- as.factor(heartfailure$Sex)
heartfailure$Smoking <- as.factor(heartfailure$Smoking)
heartfailure$Death <- as.factor(heartfailure$Death)

# Run a log Regression on each category.
logreg <- glm(Death ~ Age + Anaemia + CPK + Diabetes + `Ejection Fraction` + 
               `High Blood Pressure` + Platelets + Creatinine + Sodium + Sex + 
               Smoking + Time, 
             data = heartfailure, family = binomial)
coef(logreg)
summary(logreg)


#Data Visualizations
#Binary Variables Chart Showing Death or Not Probabilities
heartfailure %>%
  pivot_longer(cols = c(Anaemia, Diabetes, `High Blood Pressure`, Sex, Smoking), 
               names_to = "Variable", values_to = "Value") %>%
  group_by(Variable, Value, Death) %>%
  summarise(Count = n(), .groups = "drop") %>%
  group_by(Variable, Value) %>%
  mutate(Proportion = Count / sum(Count)) %>%
  ggplot(aes(x = factor(Value), y = Proportion, fill = factor(Death))) +
  geom_bar(stat = "identity", position = "fill") +
  geom_text(aes(label = round(Proportion, 2)),
            position = position_fill(vjust = 0.5),
            color = "white",
            size = 3) +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(title = "Proportion of Death by Binary Variables", 
       y = "Proportion", x = "", fill = "Death") +
  theme_minimal()

numeric_vars <- c("Age", "CPK", "Ejection Fraction", "Platelets", "Creatinine", "Sodium", "Time")

#Boxplot of Numeric Variables by Death
heartfailure %>%
  pivot_longer(cols = all_of(numeric_vars), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Death, y = Value, fill = Death)) +
  geom_boxplot() +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Boxplots of Numeric Variables by Death")

#The Major Differences we can see here are that as Age increased, probability of Death increased.
#As ejection fraction decreased, probability of Death increased as there is less blood pumping and causing a weaker heart.
#As Follow up Time decreases, meaning more visits to the doctor, the probability of Death increased. 
#As Sodium levels decreased, probability of Death increased. 

#PART 2

#Lasso
x <- model.matrix(Death ~ Age + Anaemia + CPK + Diabetes + `Ejection Fraction` + 
                    `High Blood Pressure` + Platelets + Creatinine + Sodium + 
                    Sex + Smoking + Time, data = heartfailure)[,-1] # Remove intercept

y <- heartfailure$Death

lasso_model <- gamlr(x, y, family = "binomial")

# Coefficients
coef(lasso_model)

# Plot
plot(lasso_model)

#Variable Interpretations (Log-Odds & Odds Change) from LASSO MODEL
#Age
#Log-odds: For every 1-year increase in age, the log-odds of death increase by 0.037, assuming all other variables are held constant.
#Odds: The odds of death increase by about 3.8% for each additional year of age. (exp(0.037) ≈ 1.038)

#CPK (Creatine Phosphokinase)
#Log-odds: For every 1-unit increase in CPK, the log-odds of death increase by 0.00013, assuming all other variables are held constant.
#Odds: The odds of death increase by about 0.013% per unit increase in CPK. (exp(0.00013) ≈ 1.00013)

#Ejection Fraction
#Log-odds: For every 1-unit increase in ejection fraction, the log-odds of death decrease by 0.064, assuming all other variables are held constant.
#Odds: The odds of death decrease by about 6.2% per unit increase. (exp(−0.064) ≈ 0.938)

#Platelets
#Log-odds: For every 1-unit increase in platelet count, the log-odds of death decrease by 0.00000032, assuming all other variables are held constant.
#Odds: The odds of death decrease by about 0.000032% per unit increase in platelets. (exp(−0.00000032) ≈ 0.99999968)

#Creatinine
#Log-odds: For every 1-unit increase in creatinine, the log-odds of death increase by 0.577, assuming all other variables are held constant.
#Odds: The odds of death increase by about 78% per unit increase in creatinine. (exp(0.577) ≈ 1.78)

#Sodium
#Log-odds: For every 1-unit increase in sodium, the log-odds of death decrease by 0.052, assuming all other variables are held constant.
#Odds: The odds of death decrease by about 5.1% per unit increase. (exp(−0.052) ≈ 0.949)

#Sex (1 = male)
#Log-odds: Being male decreases the log-odds of death by 0.290, compared to being female, assuming all other variables are held constant.
#Odds: Males have about 25.2% lower odds of death than females. (exp(−0.290) ≈ 0.748)
#May need deeper clinical context; could reflect sample-specific patterns.

#Time (Follow-up Days)
#Log-odds: For every additional day of follow-up, the log-odds of death decrease by 0.0188, assuming all other variables are held constant.
#Odds: The odds of death decrease by about 1.9% for each additional day survived. (exp(−0.0188) ≈ 0.981)

#These had no added predictive value at the selected regularization level:
#Anaemia
#Diabetes
#High Blood Pressure
#Smoking
#LASSO decided these were not contributing enough once other variables were accounted for, so they were excluded.

#Death is more likely with older age, higher creatinine,lower ejection fraction, lower sodium, and possibly higher CPK, while some clinical features like diabetes, anaemia, smoking, and high blood pressure didn’t improve prediction enough to keep.
#sodium: Lowers pump and blood flow going to the heart. Not likely to lead to heart failure

#“Each colored line represents a variable in our model. On the x-axis, we have log(lambda) — the regularization strength. As lambda increases (moving left), the model shrinks coefficients toward zero.
#The vertical dashed line is the lambda value selected by cross-validation — this is the point that offers the best predictive performance without overfitting.
#To the right of that line (smaller lambda), more variables are included with non-zero coefficients. But as we move left, only the most important predictors survive regularization — others are shrunk exactly to zero and dropped.


#We ran a Cross Validation Model. 
cv_model <- cv.gamlr(x, y, family = "binomial")
coef(cv_model)
plot(cv_model)

#seg 35
#“We ran a LASSO model to identify key predictors of death, which highlighted Age, CPK, Ejection Fraction, Platelets, Creatinine, Sodium, Sex, and Time as significant variables.

#However, when we applied cross-validated LASSO — which more rigorously selects variables by evaluating prediction performance on unseen data — the model retained only Age, Ejection Fraction, Creatinine, and Time, each with smaller effect sizes.

#This suggests that CPK, Platelets, Sodium, and Sex, while potentially informative in the full model, did not contribute reliably to prediction and were removed during cross-validation. In other words, they were not essential to accurately distinguishing between survival and death outcomes in this dataset.”


#Age, Ejection Fraction, Creatinine, and Time MOST IMPORTANT


# PART 3

#Scale Variables for PCA Analysis
heartfailure2 <- heartfailure %>%
  select(Age, CPK, `Ejection Fraction`, Platelets, Creatinine, Sodium, Time)

#Ran a PCA Analysis on the LASSO given Variables.
#PC1 - Age
#PC2 - CPK
#PC3 - Ejection Fraction
#PC4 - Platelets
#PC5 - Creatinine
#PC6 - Sodium
#PC7 - Time

scaled_data <- scale(heartfailure2)

# Perform PCA on numeric variables
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Summary of PCA
summary(pca_result)

# Visualize the first two principal components
pca_data <- as.data.frame(pca_result$x)
pca_data$Death <- heartfailure$Death  # Add Death for visualization


ggplot(pca_data, aes(x = PC1, y = PC2, color = Death)) +
  geom_point() +
  labs(title = "PCA of Heart Failure Data", x = "Principal Component 1 - Age", y = "Principal Component 2 - CPK") +
  theme_minimal()

#Smaller Age and Lower CPK Red dots, Less chance of death
#Higher age and Higher CPK blue dots, More chance of death
#Very visible that these are the most significant factors and ranked PCA 1 and PCA 2. 

ggplot(pca_data, aes(x = PC3, y = PC4, color = Death)) +
  geom_point() +
  labs(title = "PCA of Heart Failure Data", x = "Principal Component 3 - Ejection Fraction", y = "Principal Component 4 - Platelets") +
  theme_minimal()

#Mix; difficult to tell however lower Ejection Fraction Visible

ggplot(pca_data, aes(x = PC5, y = PC6, color = Death)) +
  geom_point() +
  labs(title = "PCA of Heart Failure Data", x = "Principal Component 5 - Creatinine", y = "Principal Component 6 - Sodium") +
  theme_minimal()

#Mix; both set around 0.

ggplot(pca_data, aes(x = PC6, y = PC7, color = Death)) +
  geom_point() +
  labs(title = "PCA of Heart Failure Data", x = "Principal Component 6 - Sodium", y = "Principal Component 7 - Time") +
  theme_minimal()

#Mix, both set around 0. 


# K-means clustering on PCA's
set.seed(123)
kmeans_result <- kmeans(pca_data[,1:4], centers = 2)

pca_data$Cluster <- as.factor(kmeans_result$cluster)

ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  labs(title = "Clustering on PCA Components", x = "Age", y = "CPK")

ggplot(pca_data, aes(x = PC3, y = PC4, color = Cluster)) +
  geom_point() +
  labs(title = "Clustering on PCA Components", x = "Ejection Fraction", y = "Platelets")

ggplot(pca_data, aes(x = PC5, y = PC6, color = Cluster)) +
  geom_point() +
  labs(title = "Clustering on PCA Components", x = "Creatinine", y = "Sodium")

ggplot(pca_data, aes(x = PC6, y = PC7, color = Cluster)) +
  geom_point() +
  labs(title = "Clustering on PCA Components", x = "Sodium", y = "Time")

#Clustering Visuals show minor changes for the classified data points.

#In Conclusion, after running a logistical regression from our heartfailure data,
#it came out that Age, Ejection Fraction, Creatinine, & Time were the significant factors.
#However, after running a LASSO model to initially validate the data, it came out that 
#Age, Anaemial, CPK, Ejection Fraction, Platelets, Creatinine, Sodium, Sex, and Time were significant variables to Death.
#Then, to validate the data's relationship, we created a Cross Validation that showed only
#Age, Ejection Fraction, Creatinine, & Time to be the ultimate significant variables in correlation to Death.
#Finally, a Principle Component Analysis (PCA) and Clustering was run to show the ranked variance among the key variables within the data.
