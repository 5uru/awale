import jax.numpy as jnp
from jax import jit, lax
from typing import NamedTuple, Tuple, Dict
from AwaleEnv.utils import distribute_seeds, capture_seeds, State, update_game_state
import chex
from typing_extensions import TypeAlias
import jax.random as random
from jax.random import PRNGKey

# Constantes du jeu
BOARD_SIZE = 12
SEEDS_PER_PIT = 4
PLAYER_SIDE_SIZE = 6
INITIAL_SCORE = 0
CAPTURE_REWARD_MULTIPLIER = 0.5


class AwaleJAX:
    """
    Implémentation du jeu Awale utilisant JAX pour des performances optimales.
    Le jeu suit les règles traditionnelles de l'Awale avec capture de graines.
    """

    def __init__(self):
        """
        Initialise une nouvelle partie d'Awale.
        Crée l'état initial du jeu en utilisant une clé aléatoire fixe.
        """
        self.state = self.reset(PRNGKey(0))

    @staticmethod
    def reset(key: PRNGKey) -> State:
        """
        Réinitialise ou commence une nouvelle partie.

        Args:
            key: Clé PRNGKey pour la génération de nombres aléatoires

        Returns:
            State: État initial du jeu
        """
        # Détermine aléatoirement le premier joueur (0 ou 1)
        current_player = (random.uniform(key) > 0.5).astype(jnp.int8)

        # Calcule l'espace d'actions valides pour le joueur actuel
        action_space = jnp.arange(
            current_player * PLAYER_SIDE_SIZE,
            (current_player + 1) * PLAYER_SIDE_SIZE,
            dtype=jnp.int8,
        )

        # Crée le plateau initial avec 4 graines dans chaque trou
        board = jnp.full(BOARD_SIZE, SEEDS_PER_PIT, dtype=jnp.int8)

        # Initialise les scores à zéro pour les deux joueurs
        scores = jnp.zeros(2, dtype=jnp.int8)

        return State(
            board=board,
            action_space=action_space,
            key=key,
            score=scores,
            current_player=current_player,
        )

    @staticmethod
    @jit
    def step(state: State, action: jnp.int8) -> Tuple[State, float, bool, Dict]:
        """
        Exécute une action dans le jeu.

        Args:
            state: État actuel du jeu
            action: Action choisie (index du trou)

        Returns:
            Tuple contenant:
            - Nouvel état du jeu
            - Récompense obtenue
            - Indicateur de fin de partie
            - Informations supplémentaires (gagnant)
        """
        # Collecte et distribue les graines
        seeds = state.board[action]
        board = state.board.at[action].set(0)

        # Distribution des graines
        board, final_pit = distribute_seeds(board, action, seeds)

        # Capture des graines si possible
        board, captured = capture_seeds(board, final_pit, state.current_player)

        # Mise à jour du score et calcul de la récompense initiale
        score = state.score.at[state.current_player].add(captured)
        reward = CAPTURE_REWARD_MULTIPLIER * captured

        # Mise à jour de l'état du jeu
        (
            new_board,
            new_scores,
            new_player,
            done,
            new_reward,
            winner,
            new_action_space,
        ) = update_game_state(board, score, state.current_player)

        # Ajout de la nouvelle récompense
        reward += new_reward

        # Filtrage des actions valides (trous non vides)
        valid_actions = jnp.where(board[new_action_space] > 0)[0]
        new_action_space = new_action_space[valid_actions]

        # Création du nouvel état
        new_state = State(
            board=new_board,
            action_space=new_action_space,
            key=state.key,
            score=new_scores,
            current_player=new_player,
        )

        return new_state, reward, done, {"winner": winner}

    @staticmethod
    def render(state: State) -> None:
        """
        Affiche l'état actuel du plateau de jeu dans la console.

        Args:
            state: État actuel du jeu à afficher
        """
        # Prépare les lignes du plateau pour l'affichage
        top_row = state.board[6:0:-1]  # Inverse la ligne du haut pour l'affichage
        bottom_row = state.board[6:]

        # Construction de l'affichage
        board_display = [
            f"Player 2: {state.score[1]:2d}",
            "   ┌────┬────┬────┬────┬────┬────┐",
            f"   │ {' │ '.join(f'{pit:2d}' for pit in top_row)} │",
            "───┼────┼────┼────┼────┼────┼────┤",
            f"   │ {' │ '.join(f'{pit:2d}' for pit in bottom_row)} │",
            "   └────┴────┴────┴────┴────┴────┘",
            f"Player 1: {state.score[0]:2d}",
        ]

        # Affiche le plateau
        print("\n".join(board_display))
