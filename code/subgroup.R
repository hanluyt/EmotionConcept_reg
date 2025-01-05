require('R.matlab')
require(tidyverse)
require(dplyr)
require(ggplot2)
require(rstatix)
library(philentropy)
library(cluster)
library(jsonlite)

########################################################################
######################## K means clustering ###########################
########################################################################
angry_neutral_occipital = readMat('data/ANN_mapping/angry_mu_seed100_occipital.mat')$angry.mu.seed100.occipital %>% 
  as.data.frame # [1332, 256]

############# The optimal number of clusters #############
#### elbow method ####
set.seed(123)
wcss <- numeric()  
for (k in 1:7) {  
  kmeans_result <- kmeans(angry_neutral_occipital, centers=k, nstart=25, iter.max=20)
  wcss[k] <- kmeans_result$tot.withinss  
}
ggplot() +
  geom_line(aes(x=1:7, y=wcss), color='blue', linewidth=1) +
  geom_point(aes(x=1:7, y=wcss), color='red', size=3) +
  labs(x='Number of clusters', y='WCSS') +
  scale_x_continuous(breaks=1:7) +
  theme_bw() + theme(panel.border = element_blank(), panel.grid.minor = element_blank(),
                     panel.grid.major = element_blank(), axis.line = element_line(colour = "black"),
                     text=element_text(size=13, family='Arial'))

# ggsave(filename='elbow.tiff',
#        width=1300,
#        height=1000,
#        units='px',
#        bg = "white",       
#        dpi=300)


#### silhouette method ####
silhouette_scores <- numeric()

for (k in 2:7) {
  kmeans_result <- kmeans(angry_neutral_occipital, centers=k, nstart=25, iter.max=20)
  silhouette_obj <- silhouette(kmeans_result$cluster, dist(angry_neutral_occipital))
  silhouette_scores[k-1] <- mean(silhouette_obj[, "sil_width"])
}

ggplot() +
  geom_line(aes(x=2:7, y=silhouette_scores), color='blue', linewidth=1) +
  geom_point(aes(x=2:7, y=silhouette_scores), color='red', size=3) +
  labs(x='Number of clusters', y='Average silhouette width') +
  scale_x_continuous(breaks=2:7) +
  theme_bw() + theme(panel.border = element_blank(), panel.grid.minor = element_blank(),
                     panel.grid.major = element_blank(), axis.line = element_line(colour = "black"),
                     text=element_text(size=13, family='Arial'))

# ggsave(filename='silhouette.tiff',
#        width=1300,
#        height=1000,
#        units='px',
#        bg = "white",       
#        dpi=300)


###################### stability testing ##################
num_clusters <- 2
num_repeats <- 50  

consensus_matrix <- matrix(0, nrow = nrow(angry_neutral_occipital), ncol = nrow(angry_neutral_occipital))
for (i in 1:num_repeats) {
  set.seed(i)
  cluster_result <- kmeans(angry_neutral_occipital, centers = num_clusters, nstart = 25)
  # Update consensus matrix
  for (j in 1:num_clusters) {
    cluster_members <- which(cluster_result$cluster == j)
    consensus_matrix[cluster_members, cluster_members] <- consensus_matrix[cluster_members, cluster_members] + 1
  }
}

consensus_matrix <- consensus_matrix / num_repeats 

############ T-SNE for visualization ###########
df_pca <- read_csv('data/tsne_cluster2.csv') %>% 
  convert_as_factor(cluster)
ggplot(df_pca, aes(x = PC1, y = PC2, color = as.factor(cluster))) +
  geom_point(alpha = 0.5) + xlab('') + ylab('') +
  scale_color_manual(values = c('1' = '#C4777D', '2' = '#809EC4')) +
  theme_minimal() + theme(panel.border = element_blank(), panel.grid.minor = element_blank(),
                          panel.grid.major = element_blank(), axis.text = element_blank(),
                          text=element_text(size=13, family='Arial'), 
                          legend.position = 'none')

# ggsave(filename='vis2.tiff',
#        width=1300,
#        height=1300,
#        units='px',
#        bg = "white",       
#        dpi=300)

##########################################################################
################ The demographic difference between two clusters ##############
##########################################################################
data <- read_csv('data/IMAGEN_covariate/IMAGEN_data/score_symptom_dawba_adrs.csv') %>% 
  convert_as_factor(id, cluster, sex, hand, site)

data$cluster <- ifelse(data$cluster == 0, 1, 0)
data$cluster <- as.factor(data$cluster)

# chi square test for handedness, sex, site
chi_table <- table(data$cluster, data$site)
chisq.test(chi_table)

# t test for ses
t.test(ses ~ cluster, data = data)


##########################################################################
################ The symptom difference between two clusters ##############
##########################################################################
# Load the symptom data
data <- read_csv('data/IMAGEN_covariate/IMAGEN_data/score_symptom_dawba_adrs.csv') %>% 
  convert_as_factor(id, cluster, sex, hand, site)

data$cluster <- ifelse(data$cluster == 0, 1, 0)
data$cluster <- as.factor(data$cluster)
data <- as.data.frame(lapply(data, function(x) if(is.numeric(x)) scale(x) else x))

model <- lm(adrs_sum ~ cluster + sex + ses + hand + site, data = data)
summary(model)
confint(model)

########### The proportion of case and control in two clusters ###########
data <- read_csv('data/IMAGEN_covariate/IMAGEN_data/score_symptom_dawba_adrs.csv') %>% 
  convert_as_factor(id, cluster, sex, hand, site)
data %>%
  group_by(cluster) %>%
  summarise(mean=mean(dep, na.rm = TRUE), std=sd(dep, na.rm=TRUE), n= sum(!is.na(dep)))

band <- read_csv('data/IMAGEN_covariate/IMAGEN_data/DAWBA_band.csv') %>%
  convert_as_factor(id)
band <- band[,c(1, 2, 9, 11)]
band$sp19_low <- ifelse(band$sspphband < 4, 0, 1)
band$dep19_low <- ifelse(band$sdepband < 4, 0, 1)
band$ep19_low <- ifelse(band$seatband < 4, 0, 1)

band$internal19_low <- ifelse(band$sspphband >= 4 | band$sdepband >= 4 | band$seatband >= 4, 1, 0)


data <- merge(data, band[,c(1,5,6,8)], by='id', all.x = TRUE)

# The number of cases in low-efficiency group: control: 654, case: 51
table(data[data$cluster==0,]$internal19_low) 
# The number of cases in high-efficiency group: control: 388, case: 9
table(data[data$cluster==1,]$internal19_low) 

#### chi-squared test ####
counts <- matrix(c(654, 51, 388, 9), nrow = 2, byrow = TRUE)
colnames(counts) <- c("0", "1")
rownames(counts) <- c("Cluster 0", "Cluster 1")

chi_test <- chisq.test(counts)
chi_test

##########################################################################
################ Information gain for two clusters ##############
##########################################################################
InfoGain <- function(data, num_bins=40) {
  randomdis = log2(num_bins) # random distribution
  mean_data = 2 * (data - min(data)) / (max(data) - min(data)) - 1 # normalize to [-1, 1]
  bins = cut(mean_data, breaks = num_bins) 
  counts = table(bins) # the number of data points in each bin
  probabilities = counts / sum(counts) # convert to probabilities

  entropy <- -sum(probabilities * log2(probabilities + 1e-10))  
  gain <- randomdis - entropy
  return(gain)
}

# Compute the 95%CI for the difference in information gain between two groups
InfoGain_twogroup <- function(data_C1, data_C2, n_iter = 1000){
  result <- data.frame(information1=numeric(1), information2=numeric(1), lower=numeric(1), 
                       upper=numeric(1), more=numeric(1))
  diffs = numeric(n_iter)
  info1_all = InfoGain(colMeans(data_C1))
  info2_all = InfoGain(colMeans(data_C2))
  
  for (j in 1:n_iter){
    # Sample with replacement
    idx1 = sample(nrow(data_C1), replace = TRUE)
    sample_C1 = data_C1[idx1, ]
    idx2 = sample(nrow(data_C2), replace = TRUE)
    sample_C2 = data_C2[idx2, ]

    info1_sample = InfoGain(colMeans(sample_C1))
    info2_sample = InfoGain(colMeans(sample_C2))
   
    diffs[j] = info2_sample - info1_sample # information
    
  }
  
  # 95%CI
  ci_p = quantile(diffs, c(0.025, 0.975))
  result[1,] <- c(info1_all, info2_all, ci_p[1], ci_p[2],
                  ifelse(ci_p[1]>=0 & ci_p[2]>0, 2, ifelse(
                    ci_p[1]<0 & ci_p[2]<=0, 1, 0)))
  result_info <- list(diffs, result)
  return(result_info)
}


set.seed(123)
angry_neutral_occipital = readMat('data/ANN_mapping/angry_mu_seed100_occipital.mat')$angry.mu.seed100.occipital %>%
  as.data.frame # [1332, 256]
kmeans_result <- kmeans(angry_neutral_occipital, centers=2, nstart=25, iter.max=20)
id <- fromJSON('data/id_json/total_id_occipital.json')
id_label <- data.frame(id=id, cluster=kmeans_result$cluster)
id_label$id <- gsub('\\.npy$', '', id_label$id)
data <- cbind(id_label$cluster, angry_neutral_occipital)
colnames(data)[1] <- 'cluster'
data_C1 <- filter(data, cluster == 1) 
data_C2 <- filter(data, cluster == 2)
data_C1 <- data_C1[, -1]
data_C2 <- data_C2[, -1]


result_info <- InfoGain_twogroup(data_C1, data_C2)



                           
                           