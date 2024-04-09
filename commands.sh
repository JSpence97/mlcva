
t_max=600  # Set algorithms to run for
## Non-adaptive methods
echo "Non-adaptive: gamma=1"
echo "Computing Level Statistics"
python main_cva.py --mode level_stats --beta 0.5 --alpha 0.5 --gamma 1 --N_00 8 --L_test 5 --name non_ad_gamma_1 --t_max $t_max # Initial estimates of beta and alpha
echo "Estimating CVA Probability of Large Loss"
python main_cva.py --mode mlmc --beta 0.5 --alpha 0.5 --gamma 1 --N_00 8 --L_test 5 --name non_ad_gamma_1 --t_max $t_max # Initial estimates of beta and alpha

echo "Non-adaptive: gamma=2"
echo "Computing Level Statistics"
python main_cva.py --mode level_stats --beta 1 --alpha 1 --gamma 2 --N_00 8 --L_test 5 --name non_ad_gamma_2 --t_max $t_max
echo "Estimating CVA Probability of Large Loss"
python main_cva.py --mode mlmc --beta 1 --alpha 1 --gamma 2 --N_00 8 --L_test 5 --name non_ad_gamma_2 --t_max $t_max

# Adaptive
echo "Adaptive: gamma=1, R = 1.95"
echo "Computing Level Statistics"
python main_cva.py --mode level_stats --beta 1 --alpha 1 --gamma 1 --N_00 8 --L_test 5 --R 1.95 --name ad_gamma_1_R_1_95 --t_max $t_max
echo "Estimating CVA Probability of Large Loss"
python main_cva.py --mode mlmc --beta 1 --alpha 1 --gamma 1 --N_00 8 --L_test 5 --R 1.95 --name ad_gamma_1_R_1_95 --t_max $t_max
