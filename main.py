import pickle
import numpy as np
import matplotlib.pyplot as plt


def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """

    NUM_TRAIN_GAMES = 1
    NUM_EXPERIMENTS = 100

    # error rates, True risks of the prophets
    first_prophet, second_prophet = 0.2, 0.4
    r_best = min(first_prophet, second_prophet)
    # Approximation error is constant for all experiments:
    approximation_error = r_best
    # read binary the pickle file
    with open("scenario_one_and_two_prophets.pkl", "rb") as f:
        scenario = pickle.load(f)
    # the true outcomes (Y):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # Y_train : the true outcomes for the 10000 training games:
    Y_train = data['train_set']
    # the true outcomes for the 1000 test games:
    Y_test = data['test_set']

    P_train = scenario['train_set']
    P_test = scenario['test_set']

    #initilize arrays to store results:
    selected_prophets_errors = np.zeros(NUM_EXPERIMENTS)
    estimation_errors = np.zeros(NUM_EXPERIMENTS)
    P1_selected_count = 0

    # THE EXPERIMENT:
    for i in range(NUM_EXPERIMENTS):
        train_game_index = np.random.choice(len(Y_train), size = NUM_TRAIN_GAMES, replace = False)
        # the true outcome and the prophets' predictions for the chosen game:
        y_sample = Y_train[train_game_index]
        p_sample = P_train[:, train_game_index]

        # CALCULATE ERM: empirical risk = (Number of mistakes) / (number of games)
        # boolean values: sum of the XOR is the number of mistakes
        erm_p1 = np.sum(p_sample[0] != y_sample) / NUM_TRAIN_GAMES
        erm_p2 = np.sum(p_sample[1] != y_sample) / NUM_TRAIN_GAMES
        # P1 SELECTED
        if erm_p1 < erm_p2:
            selected_prophet_index = 0
            R_selected_true = first_prophet
        # prophet 2 is selected:
        elif erm_p2 < erm_p1:
            selected_prophet_index = 1
            R_selected_true = second_prophet
        else: # there is a tie:
            selected_prophet_index = np.random.choice([0, 1], size=1)[0]
            if selected_prophet_index == 0:
                R_selected_true = first_prophet
            else:
                R_selected_true = second_prophet

        # track how many times the first prophet was selected
        if selected_prophet_index == 0:
            P1_selected_count += 1

        # evaluate selected prophet on the test set:
        test_predictions = P_test[selected_prophet_index]
        test_error = np.sum(test_predictions != Y_test) / len(Y_test)
        selected_prophets_errors[i] = test_error

        # calc the estimation error : R(h_selected) - r_best
        estimation_errors[i] = R_selected_true - r_best

    # --- results ---
    # Average error of the selected prophets over the 100 experiments:
    mean_selected_error = np.mean(selected_prophets_errors)
    # Average estimation error over the 100 experiments:
    mean_estimation_error = np.mean(estimation_errors)

    print(f' True Risk of the Best Prophet: {r_best:.4f}')
    print(f'Approximation Error (r_best): {approximation_error:.4f}')
    print(f'------------------------------------------------------------')
    print(f'Average Error of Selected Prophets on Test Set: {mean_selected_error:.4f}')
    print(f'Times Best Prophet (P1) was chosen: {P1_selected_count} out of 100')
    print(f'Mean Estimation Error: {mean_estimation_error:.4f}')
    print(f'------------------------------------------------------------')

    # You would also include this analysis in your report:
    print(f'\nAnalysis for Scenario 1:')
    print(
        f'The Approximation Error is constant at {approximation_error}, representing the inherent limit of our hypothesis class (2 prophets).')
    print(f'The Estimation Error is high because training on only 1 game provides very little information.')
    print(
        f'Since the train set size (m=1) is tiny, the Empirical Risk (ERM) on this set is a very poor estimate of the True Risk, leading to frequent selection of the worse prophet (P2) and a high mean estimation error.')
    print(f'-----------------------------------------------------------------')

def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """

    NUM_TRAIN_GAMES = 10
    NUM_EXPERIMENTS = 100

    # error rates, True risks of the prophets
    first_prophet, second_prophet = 0.2, 0.4
    r_best = min(first_prophet, second_prophet)
    # Approximation error is constant for all experiments:
    approximation_error = r_best
    # read binary the pickle file
    with open("scenario_one_and_two_prophets.pkl", "rb") as f:
        scenario = pickle.load(f)
    # the true outcomes (Y):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # Y_train : the true outcomes for the 10000 training games:
    Y_train = data['train_set']
    # the true outcomes for the 1000 test games:
    Y_test = data['test_set']

    P_train = scenario['train_set']
    P_test = scenario['test_set']

    #initilize arrays to store results:
    selected_prophets_errors = np.zeros(NUM_EXPERIMENTS)
    estimation_errors = np.zeros(NUM_EXPERIMENTS)
    P1_selected_count = 0

    # THE EXPERIMENT:
    for i in range(NUM_EXPERIMENTS):
        train_game_index = np.random.choice(len(Y_train), size = NUM_TRAIN_GAMES, replace = False)
        # the true outcome and the prophets' predictions for the chosen game:
        y_sample = Y_train[train_game_index]
        p_sample = P_train[:, train_game_index]

        # CALCULATE ERM: empirical risk = (Number of mistakes) / (number of games)
        # boolean values: sum of the XOR is the number of mistakes
        erm_p1 = np.sum(p_sample[0] != y_sample) / NUM_TRAIN_GAMES
        erm_p2 = np.sum(p_sample[1] != y_sample) / NUM_TRAIN_GAMES
        # P1 SELECTED
        if erm_p1 < erm_p2:
            selected_prophet_index = 0
            R_selected_true = first_prophet
        # prophet 2 is selected:
        elif erm_p2 < erm_p1:
            selected_prophet_index = 1
            R_selected_true = second_prophet
        else: # there is a tie:
            selected_prophet_index = np.random.choice([0, 1], size=1)[0]
            if selected_prophet_index == 0:
                R_selected_true = first_prophet
            else:
                R_selected_true = second_prophet

        # track how many times the first prophet was selected
        if selected_prophet_index == 0:
            P1_selected_count += 1

        # evaluate selected prophet on the test set:
        test_predictions = P_test[selected_prophet_index]
        test_error = np.sum(test_predictions != Y_test) / len(Y_test)
        selected_prophets_errors[i] = test_error

        # calc the estimation error : R(h_selected) - r_best
        estimation_errors[i] = R_selected_true - r_best

    # --- results ---
    # Average error of the selected prophets over the 100 experiments:
    mean_selected_error = np.mean(selected_prophets_errors)
    # Average estimation error over the 100 experiments:
    mean_estimation_error = np.mean(estimation_errors)

    print(f'True Risk of the Best Prophet: {r_best:.4f}')
    print(f'Approximation Error (R_best): {approximation_error:.4f}')
    print(f'------------------------------------------------------------')
    print(f'Average Error of Selected Prophets on Test Set: {mean_selected_error:.4f}')
    print(f'Times Best Prophet was chosen: {P1_selected_count} out of 100')
    print(f'Mean Estimation Error: {mean_estimation_error:.4f}')
    print(f'------------------------------------------------------------')

    # You would also include this analysis in your report:
    print(f'\nAnalysis for Scenario 2:')
    print(
        f'The Approximation Error is constant at {approximation_error}, representing the inherent limit of our hypothesis class (2 prophets).')



def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    NUM_TRAIN_GAMES = 10
    NUM_EXPERIMENTS = 100


    # read binary the pickle file
    with open("scenario_three_and_four_prophets.pkl", "rb") as f:
        scenario = pickle.load(f)
    R_true_risks = scenario['true_risk']
    R_best = np.min(R_true_risks)
    approximation_error = R_best
    # the true outcomes (Y):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # Y_train : the true outcomes for the 10000 training games:
    Y_train = data['train_set']
    # the true outcomes for the 1000 test games:
    Y_test = data['test_set']

    P_train = scenario['train_set']
    P_test = scenario['test_set']

    # initilize arrays to store results:
    selected_prophets_errors = np.zeros(NUM_EXPERIMENTS)
    estimation_errors = np.zeros(NUM_EXPERIMENTS)

    best_true_risk_indices = np.where(R_true_risks == R_best)[0]
    best_prophet_selected_count = 0
    not_1_percent_worse_count = 0

    # THE EXPERIMENT:
    for i in range(NUM_EXPERIMENTS):
        train_game_index = np.random.choice(len(Y_train), size=NUM_TRAIN_GAMES, replace=False)
        # the true outcome and the prophets' predictions for the chosen game:
        y_sample = Y_train[train_game_index]
        p_sample = P_train[:, train_game_index]

        # CALCULATE ERM: empirical risk = (Number of mistakes) / (number of games)
        # boolean values: sum of the XOR is the number of mistakes
        mistakes = p_sample != y_sample
        erm_all_prophets = np.sum(mistakes, axis = 1) / NUM_TRAIN_GAMES
        # take the best prophet:
        min_erm = np.min(erm_all_prophets)
        best_prophet_indices = np.where(erm_all_prophets == min_erm)[0]
        selected_prophets_index = np.random.choice(best_prophet_indices, size = 1)[0]
        R_selected_true = R_true_risks[selected_prophets_index]

        if selected_prophets_index in best_true_risk_indices: # check for best prophets
            best_prophet_selected_count += 1
        # count the number of prophets where they were not 1% worse than the best one
        if R_selected_true <= R_best * 1.01:
            not_1_percent_worse_count += 1

        # evaluate selected prophet on the test set:
        test_predictions = P_test[selected_prophets_index]
        test_error = np.sum(test_predictions != Y_test) / len(Y_test)
        selected_prophets_errors[i] = test_error

        # calc the estimation error : R(h_selected) - r_best
        estimation_errors[i] = R_selected_true - R_best

    # --- results ---
    # Average error of the selected prophets over the 100 experiments:
    mean_selected_error = np.mean(selected_prophets_errors)
    # Average estimation error over the 100 experiments:
    mean_estimation_error = np.mean(estimation_errors)

    print(f'True Risk of the Best Prophet: {R_best:.4f}')
    print(f'Approximation Error (R_best): {approximation_error:.4f}')
    print(f'------------------------------------------------------------')
    print(f'Average Error of Selected Prophets on Test Set: {mean_selected_error:.4f}')
    print(f'Times Best Prophet was chosen: {best_prophet_selected_count} out of 100')
    print(f'Mean Estimation Error: {mean_estimation_error:.4f}')
    print(f'------------------------------------------------------------')


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    NUM_TRAIN_GAMES = 1000
    NUM_EXPERIMENTS = 100

    # read binary the pickle file
    with open("scenario_three_and_four_prophets.pkl", "rb") as f:
        scenario = pickle.load(f)
    R_true_risks = scenario['true_risk']
    R_best = np.min(R_true_risks)
    approximation_error = R_best
    # the true outcomes (Y):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # Y_train : the true outcomes for the 10000 training games:
    Y_train = data['train_set']
    # the true outcomes for the 1000 test games:
    Y_test = data['test_set']

    P_train = scenario['train_set']
    P_test = scenario['test_set']

    # initilize arrays to store results:
    selected_prophets_errors = np.zeros(NUM_EXPERIMENTS)
    estimation_errors = np.zeros(NUM_EXPERIMENTS)

    best_true_risk_indices = np.where(R_true_risks == R_best)[0]
    best_prophet_selected_count = 0
    not_1_percent_worse_count = 0

    # THE EXPERIMENT:
    for i in range(NUM_EXPERIMENTS):
        train_game_index = np.random.choice(len(Y_train), size=NUM_TRAIN_GAMES, replace=False)
        # the true outcome and the prophets' predictions for the chosen game:
        y_sample = Y_train[train_game_index]
        p_sample = P_train[:, train_game_index]

        # CALCULATE ERM: empirical risk = (Number of mistakes) / (number of games)
        # boolean values: sum of the XOR is the number of mistakes
        mistakes = p_sample != y_sample
        erm_all_prophets = np.sum(mistakes, axis=1) / NUM_TRAIN_GAMES
        # take the best prophet:
        min_erm = np.min(erm_all_prophets)
        best_prophet_indices = np.where(erm_all_prophets == min_erm)[0]
        selected_prophets_index = np.random.choice(best_prophet_indices, size=1)[0]
        R_selected_true = R_true_risks[selected_prophets_index]

        if selected_prophets_index in best_true_risk_indices:  # check for best prophets
            best_prophet_selected_count += 1
        # count the number of prophets where they were not 1% worse than the best one
        if R_selected_true <= R_best * 1.01:
            not_1_percent_worse_count += 1

        # evaluate selected prophet on the test set:
        test_predictions = P_test[selected_prophets_index]
        test_error = np.sum(test_predictions != Y_test) / len(Y_test)
        selected_prophets_errors[i] = test_error

        # calc the estimation error : R(h_selected) - r_best
        estimation_errors[i] = R_selected_true - R_best

    # --- results ---
    # Average error of the selected prophets over the 100 experiments:
    mean_selected_error = np.mean(selected_prophets_errors)
    # Average estimation error over the 100 experiments:
    mean_estimation_error = np.mean(estimation_errors)

    print(f'True Risk of the Best Prophet: {R_best:.4f}')
    print(f'Approximation Error (R_best): {approximation_error:.4f}')
    print(f'------------------------------------------------------------')
    print(f'Average Error of Selected Prophets on Test Set: {mean_selected_error:.4f}')
    print(f'Times Best Prophet was chosen: {best_prophet_selected_count} out of 100')
    print(f'Mean Estimation Error: {mean_estimation_error:.4f}')
    print(f'------------------------------------------------------------')


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """

    NUM_EXPERIMENTS = 100
    K_PROPHETS = np.array([2, 5, 10, 50])
    M_GAMES = np.array([1, 10, 50, 1000])
    LEN_K = len(K_PROPHETS)
    LEN_M = len(M_GAMES)

    # final results tables initialize:
    mean_error_table = np.zeros((LEN_K, LEN_M))
    approximation_error_table = np.zeros((LEN_K, LEN_M))
    mean_estimation_error_table = np.zeros((LEN_K, LEN_M))

    with open("scenario_five_prophets.pkl", "rb") as f:
        scenario = pickle.load(f)

    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    Y_train = data['train_set']
    Y_test = data['test_set']
    P_train = scenario['train_set']
    P_test = scenario['test_set']

    r_true_risks = scenario['true_risk']

    for index_k, k in enumerate(K_PROPHETS):
        # choose k random prophets:
        hypo_k_indices = np.random.choice(len(r_true_risks), size=k, replace=False)
        r_k_risks = r_true_risks[hypo_k_indices]
        r_best = np.min(r_k_risks) # the best with the min. risk

        for index_m, m in enumerate(M_GAMES):
            #arrays for results:
            selected_prophets_errors = np.zeros(NUM_EXPERIMENTS)
            estimation_errors = np.zeros(NUM_EXPERIMENTS)

            # the experiment:
            for i in range(NUM_EXPERIMENTS):
                # k number of prophets, m number of games:
                train_game_index = np.random.choice(len(Y_train), size=m, replace=False)
                y_sample = Y_train[train_game_index]

                p_sample = P_train[hypo_k_indices[:, None], train_game_index]
                # ERM calc.
                mistakes = p_sample != y_sample
                erm_all_prophets = np.sum(mistakes, axis=1) / m
                #
                min_erm = np.min(erm_all_prophets)
                best_prophet_indices = np.where(erm_all_prophets == min_erm)[0]
                selected_prophets_index = np.random.choice(best_prophet_indices, size=1)[0]
                r_selected_true = r_k_risks[selected_prophets_index]

                # test set predictions:
                selected_global_index = hypo_k_indices[selected_prophets_index]
                test_predictions = P_test[selected_global_index]
                test_error = np.sum(test_predictions != Y_test) / len(Y_test)
                # saving results:
                estimation_errors[i] = r_selected_true - r_best
                selected_prophets_errors[i] = test_error

            # saving the mean in a table:
            mean_selected_error = np.mean(selected_prophets_errors)
            mean_estimation_error = np.mean(estimation_errors)

            mean_error_table[index_k, index_m] = mean_selected_error
            approximation_error_table[index_k, index_m] = r_best
            mean_estimation_error_table[index_k, index_m] = mean_estimation_error

    print("-------------------- scenerio 5 -----------------------")
    print("Mean error table")
    print(mean_error_table)
    print("approximation error table")
    print(approximation_error_table)
    print("mean estimation error")
    print(mean_estimation_error_table)


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    with open ("scenario_six_prophets.pkl", "rb") as s:
        scenario = pickle.load(s)
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    hypo_1 = scenario["hypothesis1"]
    hypo_2 = scenario["hypothesis2"]

    NUM_TRAIN = 10
    TEST_GAMES = 1000

    # calc the error of the 2 prophets
    NUM_EXPERIMENTS = 100

    R_true_risks_h1 = hypo_1["true_risk"]
    R_true_risks_h2 = hypo_2["true_risk"]

    R_best_h1  = np.min(R_true_risks_h1)
    R_best_h2 = np.min(R_true_risks_h2)

    # the true outcomes (Y):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # Y_train : the true outcomes for the 10000 training games:
    Y_train = data['train_set']
    # the true outcomes for the 1000 test games:
    Y_test = data['test_set']

    P_train_h1 = hypo_1['train_set']
    P_train_h2 = hypo_2['train_set']

    P_test_h1 = hypo_1['test_set']
    P_test_h2 = hypo_2['test_set']

    # initialize arrays to store results:
    selected_prophets_errors_h1 = np.zeros(NUM_EXPERIMENTS)
    selected_prophets_errors_h2 = np.zeros(NUM_EXPERIMENTS)

    estimation_errors_h1 = np.zeros(NUM_EXPERIMENTS)
    estimation_errors_h2 = np.zeros(NUM_EXPERIMENTS)

    best_true_risk_indices_h1 = np.where(R_true_risks_h1 == R_best_h1)[0]
    best_true_risk_indices_h2 = np.where(R_true_risks_h2 == R_best_h2)[0]

    best_prophet_selected_count_h1 = 0
    best_prophet_selected_count_h2 = 0

    # THE EXPERIMENT:
    for i in range(NUM_EXPERIMENTS):
        train_game_index = np.random.choice(len(Y_train), size=NUM_TRAIN, replace=False)
        # the true outcome and the prophets' predictions for the chosen game:
        y_sample = Y_train[train_game_index]

        p_sample_h1 = P_train_h1[:, train_game_index]
        p_sample_h2 = P_train_h2[:, train_game_index]

        # CALCULATE ERM: empirical risk = (Number of mistakes) / (number of games)
        # boolean values: sum of the XOR is the number of mistakes
        mistakes_h1  = p_sample_h1 != y_sample
        mistakes_h2 = p_sample_h2 != y_sample

        erm_all_prophets_h1 = np.sum(mistakes_h1, axis=1) / NUM_TRAIN
        erm_all_prophets_h2 = np.sum(mistakes_h2, axis=1) / NUM_TRAIN

        # take the best prophet:
        min_erm_h1 = np.min(erm_all_prophets_h1)
        min_erm_h2 = np.min(erm_all_prophets_h2)

        best_prophet_indices_h1 = np.where(erm_all_prophets_h1 == min_erm_h1)[0]
        best_prophet_indices_h2 = np.where(erm_all_prophets_h2 == min_erm_h2)[0]

        selected_prophets_index_h1 = np.random.choice(best_prophet_indices_h1, size=1)[0]
        selected_prophets_index_h2 = np.random.choice(best_prophet_indices_h2, size=1)[0]

        R_selected_true_h1 = R_true_risks_h1[selected_prophets_index_h1]
        R_selected_true_h2 = R_true_risks_h2[selected_prophets_index_h2]

        if selected_prophets_index_h1 in best_true_risk_indices_h1:  # check for best prophets
            best_prophet_selected_count_h1 += 1
        if selected_prophets_index_h2 in best_true_risk_indices_h2:
            best_prophet_selected_count_h2 +=1


        # evaluate selected prophet on the test set:
        test_predictions_h1 = P_test_h1[selected_prophets_index_h1]
        test_predictions_h2 = P_test_h2[selected_prophets_index_h2]

        test_error_h1 = np.sum(test_predictions_h1 != Y_test) / len(Y_test)
        test_error_h2 = np.sum(test_predictions_h2 != Y_test) / len(Y_test)

        selected_prophets_errors_h1[i] = test_error_h1
        selected_prophets_errors_h2[i] = test_error_h2

        # calc the estimation error : R(h_selected) - r_best
        estimation_errors_h1[i] = R_selected_true_h1 - R_best_h1
        estimation_errors_h2[i] = R_selected_true_h2 - R_best_h2

    # --- results ---
    # Average error of the selected prophets over the 100 experiments:
    mean_selected_error_h1 = np.mean(selected_prophets_errors_h1)
    mean_selected_error_h2 = np.mean(selected_prophets_errors_h2)

    # Average estimation error over the 100 experiments:
    mean_estimation_error_h1 = np.mean(estimation_errors_h1)
    mean_estimation_error_h2 = np.mean(estimation_errors_h2)

    print(" ---------------- scenerio 6 ---------------------")
    print("mean selected error h1")
    print(mean_selected_error_h1)
    print("mean selected error h2")
    print(mean_selected_error_h2)

    print("mean estimation error h1")
    print(mean_estimation_error_h1)
    print("mean estimation error h2")
    print(mean_estimation_error_h2)

    # קביעת טווח ותאים משותפים
    max_error = max(np.max(estimation_errors_h1), np.max(estimation_errors_h2))
    bins = np.linspace(0, max_error + 0.01, 20)

    plt.figure(figsize=(10, 6))

    plt.hist(estimation_errors_h1, bins=bins, alpha=0.6, label='H1', density=True, color='skyblue')
    plt.hist(estimation_errors_h2, bins=bins, alpha=0.6, label='H2', density=True, color='salmon')

    plt.title(' (Estimation Errors)')
    plt.xlabel('Estimation error ($R(h_{\\text{selected}}) - R_{\\text{best}}$)')
    plt.ylabel('density')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    plt.savefig('scenario_6_estimation_histogram.png')

    return (mean_selected_error_h1, mean_selected_error_h2, R_best_h1, R_best_h2, mean_estimation_error_h1,
            mean_estimation_error_h2)



def pac_learning_analysis_1():
        hypo_class = 100
        confidence_level = 0.01
        desired_accuracy_min = 0.01
        desired_accuracy_max = 0.2

        # graph calc:
        # m >= 2 * log (2*H / confi.) / (e^2)

        C = 2 * np.log(2*hypo_class / confidence_level)
        epsilon_values =  np.linspace(desired_accuracy_min, desired_accuracy_max, 500)
        m_values =np.ceil( C / (epsilon_values ** 2))

        plt.figure(figsize=(10, 6))  # הגדרת גודל הגרף
        plt.plot(epsilon_values, m_values, color='blue', label=f'hypo. class size={hypo_class}, $\\delta$={confidence_level}')

        plt.title(
            'Minimal Samples Required (n) vs. Desired Accuracy ($\\epsilon$)\nPAC Learning: $n \\propto 1/\\epsilon^2$',
            fontsize=16)
        plt.xlabel('Desired Accuracy ($\\epsilon$ - Estimation Error)', fontsize=12)
        plt.ylabel('Minimal Samples Required (n)', fontsize=12)

        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.legend()

        # הצגת הגרף
        plt.savefig('pac_learning_1.png')


def pac_learning_analysis_2():
    hypo_class = 100
    confidence_level = 0.05
    desired_accuracy_min = 10**-4
    desired_accuracy_max = 0.1

    # graph calc:
    # m >= 2 * log (2*H / confi.) / (e^2)

    C = 2 * np.log(2 * hypo_class / confidence_level)
    epsilon_values = np.linspace(desired_accuracy_min, desired_accuracy_max, 500)
    m_values = np.ceil(C / (epsilon_values ** 2))

    plt.figure(figsize=(10, 6))  # הגדרת גודל הגרף
    plt.plot(epsilon_values, m_values, color='blue',
             label=f'hypo. class size={hypo_class}, $\\delta$={confidence_level}')

    plt.yscale('log')

    # הוספת כותרות וסימונים
    plt.title('Minimal Samples Required (n) vs. Desired Accuracy ($\\epsilon$)\nLogarithmic Y-Axis', fontsize=16)
    plt.xlabel('Desired Accuracy ($\\epsilon$)', fontsize=12)
    plt.ylabel('Minimal Samples Required (n) (Log Scale)', fontsize=12)

    # הוספת רשת
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()

    # שמירת הגרף
    plt.savefig('pac_learning_2.png')


if __name__ == '__main__':
    

    print(f'Scenario 1 Results:')
    Scenario_1()

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()


    pac_learning_analysis_1();
    pac_learning_analysis_2();
