from jax import numpy as jnp


def get_action_space(board, player_id):
    """
    Détermine l'ensemble des actions valides pour un joueur donné.

    Args:
        board : État actuel du plateau
        player_id (int): ID du joueur (0 ou 1)

    Returns:
         Liste des indices des trous jouables
    """
    valid_actions = jnp.zeros(12, dtype=jnp.int8)
    player_id = int(player_id)  # Convert player_id to a standard Python integer
    start_pit = player_id * 6
    end_pit = start_pit + 6

    # Vérifier si l'adversaire n'a pas de graines
    opponent_side = board[6:12] if player_id == 0 else board[:6]
    opponent_empty = sum(opponent_side) == 0

    for pit in range(start_pit, end_pit):
        seeds = board[pit]

        # Ne pas inclure les trous vides
        if seeds == 0:
            continue

        # Vérifier si ce coup peut nourrir l'adversaire quand nécessaire
        if opponent_empty:
            # Calculer si les graines atteignent le camp adverse
            distance_to_opponent = (12 - pit) if player_id == 0 else (6 - pit)
            if seeds > distance_to_opponent:
                valid_actions = valid_actions.at[pit].set(1)
        else:
            # Si l'adversaire a des graines, tous les coups non vides sont valides
            valid_actions = valid_actions.at[pit].set(1)

        # Vérification supplémentaire pour éviter les coups qui font plus d'un tour complet
        if seeds > 11:
            # S'assurer que ce n'est pas un coup qui capture toutes les graines adverses
            test_board = board.clone()
            test_board, captured = distribute_seeds(test_board, pit)
            opponent_after = sum(test_board[6:12] if player_id == 0 else test_board[:6])
            if opponent_after == 0:
                valid_actions[pit] = 0

    return valid_actions


def distribute_seeds(board, pit_index) -> [jnp.array, jnp.int8]:
    # Récupérer les graines du trou sélectionné
    seeds_to_distribute = board[pit_index]
    board = board.at[pit_index].set(0)

    if seeds_to_distribute == 0:
        return board, 0

    current_pit = pit_index
    captured_seeds = 0

    # Distribution des graines
    while seeds_to_distribute > 0:
        current_pit = (current_pit + 1) % 12
        # Ne pas redistribuer dans le trou de départ
        if current_pit != pit_index:
            board = board.at[current_pit].add(1)
            seeds_to_distribute -= 1

    # Vérification des captures (côté opposé seulement)
    current_player_side = pit_index // 6
    opposite_side = 1 - current_player_side

    # On capture si la dernière graine fait 2 ou 3.
    while current_pit // 6 == opposite_side and board[current_pit] in [2, 3]:
        captured_seeds += board[current_pit]
        board = board.at[current_pit].set(0)
        if current_pit == 0:
            break
        current_pit -= 1

    return board, captured_seeds


def determine_game_over(board, scores):
    """
    Détermine si le jeu est terminé et qui est le gagnant.

    Args:
        board : État actuel du plateau
        scores : Scores des joueurs

    Returns:
        tuple: (is_game_over, winner, reason)
            - is_game_over (bool): True si le jeu est terminé
            - winner (int or None): 0 pour joueur 1, 1 pour joueur 2, None pour égalité
            - reason (str): Raison de la fin du jeu
    """
    total_seeds_on_board = sum(board)
    total_captured = sum(scores)

    # Vérifier si un joueur a plus de la moitié des graines
    if scores[0] > 24:
        return True, 0, "Joueur 1 a capturé la majorité des graines"
    if scores[1] > 24:
        return True, 1, "Joueur 2 a capturé la majorité des graines"

    # Vérifier s'il ne reste plus de graines sur le plateau
    if total_seeds_on_board == 0:
        if scores[0] > scores[1]:
            return True, 0, "Plus de graines - Joueur 1 gagne"
        elif scores[1] > scores[0]:
            return True, 1, "Plus de graines - Joueur 2 gagne"
        else:
            return True, None, "Match nul"

    # Vérifier si un joueur n'a plus de graines dans son camp
    player_1_seeds = sum(board[:6])
    player_2_seeds = sum(board[6:12])

    # Vérifier si un joueur ne peut plus nourrir l'autre
    def can_feed_opponent(board, player_side):
        start = player_side * 6
        end = (player_side + 1) * 6
        opponent_empty = sum(board[1 - player_side * 6 : (2 - player_side) * 6]) == 0

        if opponent_empty:
            # Vérifier si au moins un coup peut donner des graines à l'adversaire
            for i in range(start, end):
                if board[i] > (
                    11 - i
                ):  # Assez de graines pour atteindre le camp adverse
                    return True
        return False

    # Si un joueur ne peut pas nourrir l'autre et que c'est son tour
    if player_1_seeds == 0 or (not can_feed_opponent(board, 0) and player_2_seeds == 0):
        remaining_seeds = total_seeds_on_board
        player_1_total = scores[0] + remaining_seeds
        player_2_total = scores[1]

        if player_1_total > player_2_total:
            return True, 0, "Impossibilité de nourrir - Joueur 1 gagne"
        elif player_2_total > player_1_total:
            return True, 1, "Impossibilité de nourrir - Joueur 2 gagne"
        else:
            return True, None, "Match nul - Impossibilité de nourrir"

    # Le jeu continue
    return False, None, "Jeu en cours"


def calculate_reward(board, captured_seeds, player_id, game_over=False):
    """
    Calcule la récompense pour un état donné du jeu d'awalé.

    Args:
        board : État actuel du plateau
        captured_seeds : Nombre de graines capturées dans le dernier coup
        player_id : ID du joueur (0 ou 1)
        game_over : Indique si le jeu est terminé

    Returns:
        jnp.float16: La récompense calculée
    """
    # Points de base pour la capture de graines
    reward = captured_seeds * 2.0

    # Récompense/pénalité pour fin de partie
    if game_over:
        total_seeds = sum(board)
        if total_seeds == 0:
            player_score = captured_seeds
            if player_score > 24:  # Plus de la moitié des graines
                reward += 50.0  # Bonus de victoire
            elif player_score == 24:  # Égalité
                reward += 10.0
            else:
                reward -= 50.0  # Pénalité de défaite

    # Convert player_id to a standard Python integer
    player_id = int(player_id)

    # Évaluation de la position stratégique
    player_side = board[player_id * 6 : (player_id + 1) * 6]
    opponent_side = board[(1 - player_id) * 6 : (2 - player_id) * 6]

    # Bonus pour maintenir des graines de son côté
    reward += sum(player_side) * 0.1

    # Bonus pour les positions permettant des captures futures
    potential_captures = sum(seeds in [2, 3] for seeds in opponent_side)
    reward += potential_captures * 0.5

    # Pénalité pour les positions vulnérables
    vulnerable_positions = sum(seeds in [2, 3] for seeds in player_side)
    reward -= vulnerable_positions * 0.3

    # Bonus pour le contrôle du centre (trous 2-4)
    center_control = sum(player_side[2:5]) * 0.2
    reward += center_control

    # Pénalité pour les coups qui laissent l'adversaire sans coup possible
    if all(seeds == 0 for seeds in opponent_side):
        reward -= 40.0

    return jnp.float16(reward)
