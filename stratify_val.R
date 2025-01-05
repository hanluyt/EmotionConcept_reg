require('R.matlab')
require(tidyverse)
require(dplyr)
require(ggplot2)
require(rstatix)
library(philentropy)
library(cluster)
library(jsonlite)

#############################################
###### brain score(stratify, CRPN) ##########
#############################################
data <- read_csv('/Users/user/emotionANN/data/stratify_activation/score_symptom_stratify.csv') %>% 
  convert_as_factor(id, Sex, Site, Group)

data_mdd <- data %>% filter(Group == 'MDD') # 134
data_control <- data %>% filter(Group == 'Control') # 62
data_mdd_control <- rbind(data_mdd, data_control)
data_mdd_control$Group <- factor(data_mdd_control$Group)

data_AN <- data %>% filter(Group == 'AN') # 52
data_AUD <- data %>% filter(Group == 'AUD') # 122
data_BN <- data %>% filter(Group == 'BN') # 41

data_an_control <- rbind(data_AN, data_control)
data_an_control$Group <- factor(data_an_control$Group)

data_aud_control <- rbind(data_AUD, data_control)
data_aud_control$Group <- factor(data_aud_control$Group)

data_bn_control <- rbind(data_BN, data_control)
data_bn_control$Group <- factor(data_bn_control$Group)


model_high <- lm(lamdahigh ~ Group + Sex + Site + age, data=data_mdd_control)
summary(model_high)

data_mdd_control %>% 
 group_by(Group) %>% 
 summarise(mean = mean(lamdahigh), sd = sd(lamdahigh), n = n())





