require('R.matlab')
require(tidyverse)
require(dplyr)
require(ggplot2)
require(rstatix)
library(philentropy)
library(cluster)
library(jsonlite)
options(warn = -1)

#################################################################################
#### In silico experiments on ANNs to simulate varying levels of information #####
#################################################################################
info <- read_csv('data/ANN_bit/info_105group.csv') %>% 
  convert_as_factor(group, seed)

######### Pairwise comparison of information  #####
anova_result <- aov(information ~ group, data = info)
summary(anova_result)
eta_sq <- partial_eta_squared(anova_result)

mat <- as.data.frame(TukeyHSD(anova_result)$group)
mat$compare <- rownames(mat)
mat <- as.tibble(mat)
colnames(mat)[4] <- 'p_adj'
mat <- mat[, -c(2, 3)]
mat <- mat %>%
  mutate(signif = case_when(
    p_adj < 0.001 ~ "***",
    p_adj < 0.01 ~ "**",
    p_adj < 0.05 ~ "*",
    TRUE ~ NA
  ))
mat <- mat %>%
  separate(compare, into = c("group1", "group2"), sep = "-") %>%
  mutate(group1 = as.numeric(group1), group2 = as.numeric(group2))
mat <- mat[, -2]

heatmap_data <- mat %>% 
  mutate(signif=replace_na(signif, ""))

ggplot(heatmap_data, aes(x = factor(group2), y = factor(group1), fill = diff)) +
  geom_tile(color = "white") +
  geom_text(aes(label = signif), color = "black", size = 4.5) +
  scale_fill_gradient2(low = "#7E95E3", high = "#E37E95", mid = "white", midpoint = 0) +
  theme_bw() +
  labs(x = "",
       y = "",
       fill = "") +
  theme(legend.position = 'top', 
        panel.border = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(angle = 45, hjust = 1), text=element_text(size=20, family='Arial')) +
  scale_x_discrete(expand = c(0, 0), breaks = levels(factor(heatmap_data$group2))[c(TRUE, FALSE)]) +
  scale_y_discrete(expand = c(0, 0), limits = rev(levels(factor(heatmap_data$group2))),breaks = levels(factor(heatmap_data$group2))[c(TRUE, FALSE)]) +
  coord_fixed() +
  guides(color = guide_colorbar(direction = "horizontal", title.position = "top"))


# ggsave(filename='TukeyHSD_210_11.tiff',
#        width=1600,
#        height=1600,
#        units='px',
#        bg = "white",
#        dpi=300)


mean_data <- info %>%
  group_by(group) %>%
  summarize(mean_information = mean(information))

ggplot(info, aes(x = group, y = information)) +
  geom_line(aes(group=seed, color = as.factor(seed)), linewidth=1.5) +
  geom_point(aes(color = as.factor(seed)), size=1.8) +
  scale_color_manual(values = c("#C4CEE3", "#CAC4E3", "#D9C4E3", "#E3C4CE", "#E3CAC4")) +
  stat_summary(fun = mean, geom = "point", size = 5, color = "#97999C") +
  geom_line(data = mean_data, aes(x = group, y = mean_information, group = 1), color = "#97999C", size = 4)+
  scale_y_continuous(limits=c(0.3, 0.75), breaks=seq(0.3, 0.75, 0.1)) +
  scale_x_discrete(expand = c(0, 0), breaks = levels(factor(heatmap_data$group2))[c(TRUE, FALSE)])+
  theme_bw() +
  theme(legend.position = "none", panel.border = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(), axis.line = element_line(linewidth=0.7, colour = "black"),
        text=element_text(size=25, family='Arial'),
        axis.text.x = element_text(angle = 60, hjust = 1))+
  xlab('')+ylab('')+
  annotate('segment', x = 11, xend = 11, y = 0.3, yend = max(mean_data$mean_information), linetype = 'dashed', size = 1.7, color = '#5677b5')

# ggsave(filename='ANN_information11.tiff',
#        width=2000,
#        height=1600,
#        units='px',
#        bg = "white",
#        dpi=300)


################################################################################################
###### Brain score (cluster2, CRPN(lamda2)) -  Brain score (cluster1, CRPN(lamda2))#########
################################################################################################
data <- read_csv('data/ann21_mapping.csv')
data_s1 <- data[, c(1, 2, 4, 5)]
data_s2 <- data[, c(1, 3, 6, 7)]
colnames(data_s1) <- c('group', 'score', 'low', 'high')
colnames(data_s2) <- c('group', 'score', 'low', 'high')
data_s1$type <- 's1'
data_s2$type <- 's2'
data_s1s2 <- rbind(data_s1, data_s2)


ggplot(data, aes(x=group, y=delta_s2_1)) +
  geom_vline(xintercept = 0.4, linewidth =20, color = "#f3d7c6", alpha = 0.3) +
  geom_vline(xintercept = 1.85, linewidth =28, color = "#C6CCF3", alpha = 0.3) +
  geom_vline(xintercept = 1.05, linewidth =41, color = "#F3B8C3", alpha = 0.2) +
  geom_line(color="#96989E", linewidth=1.3) +  # 绘制折线
  geom_ribbon(aes(ymin = low_CI, ymax = high_CI), fill="#96989E", alpha = 0.4) +  
  geom_hline(yintercept = 0, linewidth =1.2, linetype='twodash', color = "#5677b5") +
  scale_x_continuous(breaks=seq(0, 2, 0.2), expand = c(0, 0.01)) +
  theme_minimal() +
  labs(x = "",
       y = "") +
  theme(panel.border = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(), axis.line = element_line(size=0.5, colour = "black"),
        text=element_text(size=15, family='Arial')) +
  annotate("segment", x = 1, xend = 1, y = -0.24, yend = max(data$delta_s2_1), 
           color = "#5677b5", size = 1.2, linetype='dashed')
  

# ggsave(filename='delta_lamda.tiff',
#        width=1700,
#        height=1000,
#        units='px',
#        bg = "white",
#        dpi=300)


#############################################################################
#### High CRPN coding and optimal CRPN coding (symptom ~ brain score) ######
################# Symptom at age 19 ~ brain score at age 19 ##################
############################################################################
data <- read_csv('data/IMAGEN_covariate/IMAGEN_data/score_symptom_dawba_adrs.csv') %>% 
  convert_as_factor(id, cluster, sex, hand, site)

data <- data.frame(lapply(data, function(x) if(is.numeric(x)) scale(x) else x)) # normalize

num_var <- 4
result_high <- data.frame(beta=numeric(num_var), std=numeric(num_var), t_value=numeric(num_var), p_value=numeric(num_var))
result_low <- data.frame(beta=numeric(num_var), std=numeric(num_var), t_value=numeric(num_var), p_value=numeric(num_var))
result_mid <- data.frame(beta=numeric(num_var), std=numeric(num_var), t_value=numeric(num_var), p_value=numeric(num_var))
indx <- c(13, 14, 17, 18)
iter <- 1

for (i in indx){
  model_high <- lm(data[,i] ~ lamdahigh + sex + bmi + ses + site, data = data)
  result_high[iter,] <- summary(model_high)$coefficients[2,]
  rownames(result_high)[iter] <- colnames(data)[i]
  
  model_mid <- lm(data[,i]~ lamdamid + sex + bmi + ses + site, data = data)
  result_mid[iter,] <- summary(model_mid)$coefficients[2,]
  rownames(result_mid)[iter] <- colnames(data)[i]
  
  model_low <- lm(data[,i] ~ lamdalow + sex + bmi + ses + site, data = data)
  result_low[iter,] <- summary(model_low)$coefficients[2,]
  rownames(result_low)[iter] <- colnames(data)[i]
  iter <- iter + 1
}

result_high$significant <- ifelse(result_high$p_value >= 0.05, 'ns', ifelse(result_high$p_value < 0.001, '***', 
                                                                            ifelse(result_high$p_value < 0.01, '**', '*')))

result_low$significant <- ifelse(result_low$p_value >= 0.05, 'ns', ifelse(result_low$p_value < 0.001, '***', 
                                                                          ifelse(result_low$p_value < 0.01, '**', '*')))

result_mid$significant <- ifelse(result_mid$p_value >= 0.05, 'ns', ifelse(result_mid$p_value < 0.001, '***',
                                                                          ifelse(result_mid$p_value < 0.01, '**', '*')))

######## linear regression plot #########
data <- data[complete.cases(data),]

model_dep <- lm(internal ~ sex + hand + site + ses, data = data)
residuals_dep <- residuals(model_dep)
residuals_dep <- scale(residuals_dep)
model_lamdahigh <- lm(lamdamid ~ sex + hand + site + ses, data = data)
residuals_lamdahigh <- residuals(model_lamdahigh)
residuals_lamdahigh <- scale(residuals_lamdahigh)
residuals_data <- data.frame(residuals_dep, residuals_lamdahigh, cluster = data$cluster)
residuals_data$cluster <- ifelse(residuals_data$cluster == 0, 'cluster1', 'cluster2')

residuals_data$cluster <- factor(residuals_data$cluster, unique(residuals_data$cluster))

# #ccd3ee, #A1A4A5, #E3C2C7
# dep_high:  scale_y_continuous(breaks=seq(-1, 4, 1)) 
ggplot(residuals_data, aes(x = residuals_lamdahigh, y = residuals_dep)) +
  geom_point(color="#E3C2C7", alpha = 0.5, size=2.5) + 
  geom_smooth(method = "lm", col = "#A1A4A5", fill='#A1A4A5', linewidth=1.2) + 
  labs(y = "Emotional symptoms", x = "Optimal brain score") +
  theme_minimal() +
  theme(panel.border = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(), axis.line = element_line(linewidth=0.7, colour = "black"),
        text=element_text(size=18, family='Arial'),
        axis.text = element_text(size=15))

# ggsave(filename='emotion_19mid.tiff',
#        width=1600,
#        height=1300,
#        units='px',
#        bg = "white",
#        dpi=300)

#############################################################################
#### High CRPN coding and optimal CRPN coding (symptom ~ brain score) ######
################# Symptom at age 23 ~ brain score at age 19 ##################
############################################################################
band <- read_csv('data/IMAGEN_covariate/IMAGEN_data/DAWBA_band.csv') %>%
  convert_as_factor(id)
band <- band[,c(1, 2, 9, 11)]
band$sp19_low <- ifelse(band$sspphband < 4, 0, 1)
band$dep19_low <- ifelse(band$sdepband < 4, 0, 1)
band$ep19_low <- ifelse(band$seatband < 4, 0, 1)

band$internal19_low <- ifelse(band$sspphband >= 4 | band$sdepband >= 4 | band$seatband >= 4, 1, 0)


dawba <- read_csv('data/IMAGEN_covariate/IMAGEN_data/DAWBA_disorder_23.csv') %>%
  convert_as_factor(id)
dawba <- dawba[complete.cases(dawba),]
colnames(dawba)[2:5] <- paste0(colnames(dawba)[2:5], '23')


data <- read_csv('data/IMAGEN_covariate/IMAGEN_data/score_symptom_dawba_adrs.csv')%>% 
  convert_as_factor(id, cluster, sex, hand, site)
data <- data[,c(1:10, 13, 14,17)]
colnames(data)[11:13] <- paste0(colnames(data)[11:13], '19')

data <- merge(data, dawba, by='id')
data <- data[complete.cases(data),] # 760

data <- data[data$id %in% band[band$internal19_low==0,]$id,] # 725

model1 <- lm(internal23 ~ lamdahigh + sex + ses + bmi + site, data=data)
summary(model1)

predictions <- predict(model1)
cor.test(predictions, data$internal23)
cor1 <- cor(predictions, data$internal23)

#### baseline model ####
model2 <- lm(internal23 ~ sex + ses + hand + site , data=data)
summary(model2)
predictions <- predict(model2)
cor.test(predictions, data$internal23)
cor2 <- cor(predictions, data$internal23)

summary1 <- summary(model1)
summary2 <- summary(model2)

adj_r2_model1 <- summary1$adj.r.squared
adj_r2_model2 <- summary2$adj.r.squared

aic_model1 <- AIC(model1)
aic_model2 <- AIC(model2)

bic_model1 <- BIC(model1)
bic_model2 <- BIC(model2)




