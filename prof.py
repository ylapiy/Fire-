# ----- Agente de exemplo: BFSAgent (planeja caminho e executa ações) -----

from collections import deque
# agente_base.py
# Esqueleto do agente para o ambiente de Labirinto

class BaseAgent(Agent):
    """
    Esqueleto para os alunos:
    - Implementar select_action (obrigatório).
    - Usar observe (opcional) para atualizar sua estratégia.
    - Se quiser, sobrescrever train_episode.
    """

    def __init__(self, env: MazeEnv):
        self.env = env
        # TODO: inicialize estruturas necessárias (ex.: fila, memória, Q-Table, heurística, etc.)
        pass

    def select_action(self, state: Pos) -> int:
        """
        Recebe o estado atual (linha, coluna) e deve retornar uma ação:
          0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
        """
        # TODO: implementar lógica aqui
        # Sugestões:
        # - Escolher aleatoriamente entre ações legais
        # - Implementar uma busca (DFS, BFS, A*, etc.)
        # - Usar Q-Learning determinístico
        raise NotImplementedError("Implemente a lógica de seleção de ação aqui.")

    def observe(self, s: Pos, a: int, r: float, s_next: Pos, done: bool) -> None:
        """
        Método opcional: pode ser usado para aprender com as transições.
        s       = estado atual
        a       = ação escolhida
        r       = recompensa recebida
        s_next  = próximo estado
        done    = se o episódio terminou
        """
        # TODO: atualizar estruturas do agente se for usar aprendizado
        pass

    def train_episode(self, env: MazeEnv, max_steps: int = 500) -> float:
        """
        Loop de treino: pode ser sobrescrito se desejar um controle maior.
        Por padrão, herda o comportamento da classe Agent.
        """
        return super().train_episode(env, max_steps)


# maze_lab_deterministico.py
# Ambiente de Labirinto determinístico (paredes, armadilhas, saída).
# Os alunos devem implementar um agente herdando de Agent (métodos abstratos).

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod

Pos = Tuple[int, int]  # (linha, coluna)

ACTIONS = {
    0: (-1, 0),  # UP
    1: (0, 1),   # RIGHT
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
}
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


@dataclass(frozen=True)
class StepResult:
    next_state: Pos
    reward: float
    done: bool
    info: Dict


class MazeEnv:
    """
    Ambiente determinístico de labirinto.
    - Grade com:
        'S' = início
        'G' = objetivo (saída)
        '#' = parede
        'T' = armadilha
        '.' = livre
    - A transição é determinística: ação -> movimento. Se bater em parede/ou fora, permanece no lugar.
    - Regras de recompensa (configuráveis):
        step_cost: custo por passo
        wall_penalty: penalidade ao tentar atravessar parede/borda
        trap_penalty: penalidade ao pisar em armadilha
        goal_reward: recompensa ao alcançar a saída
    """
    def __init__(
        self,
        grid: List[str],
        step_cost: float = -1.0,
        wall_penalty: float = -2.0,
        trap_penalty: float = -5.0,
        goal_reward: float = 50.0,
        max_steps: int = 500
    ):
        self.grid_raw = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0]) if self.n_rows > 0 else 0
        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.trap_penalty = trap_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self._validate_rectangular()
        self.start = self._find_unique('S')
        self.goal = self._find_unique('G')
        self.walls = self._collect('#')
        self.traps = self._collect('T')

        self._state: Pos = self.start
        self._steps: int = 0
        self._done: bool = False

    # ---------- helpers ----------
    def _validate_rectangular(self):
        if self.n_rows == 0 or self.n_cols == 0:
            raise ValueError("Grade vazia.")
        for r in self.grid_raw:
            if len(r) != self.n_cols:
                raise ValueError("Todas as linhas da grade devem ter o mesmo comprimento.")

    def _inside(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _find_unique(self, ch: str) -> Pos:
        found: Optional[Pos] = None
        for i, row in enumerate(self.grid_raw):
            for j, v in enumerate(row):
                if v == ch:
                    if found is not None:
                        raise ValueError(f"Mais de um '{ch}' encontrado.")
                    found = (i, j)
        if found is None:
            raise ValueError(f"Nenhum '{ch}' encontrado.")
        return found

    def _collect(self, ch: str) -> set[Pos]:
        s = set()
        for i, row in enumerate(self.grid_raw):
            for j, v in enumerate(row):
                if v == ch:
                    s.add((i, j))
        return s

    def _cell(self, pos: Pos) -> str:
        r, c = pos
        return self.grid_raw[r][c]

    # ---------- API do ambiente ----------
    @property
    def state(self) -> Pos:
        return self._state

    def reset(self) -> Pos:
        """Determinístico: sempre reseta na mesma posição."""
        self._state = self.start
        self._steps = 0
        self._done = False
        return self._state

    def step(self, action: int) -> StepResult:
        """
        Executa uma ação determinística.
        - Se a nova posição for parede/borda, agente permanece no lugar e recebe wall_penalty + step_cost.
        - Se a nova posição for armadilha, aplica trap_penalty + step_cost.
        - Se for a saída, aplica goal_reward e finaliza.
        - Caso contrário, apenas step_cost.
        """
        if self._done:
            return StepResult(self._state, 0.0, True, {"msg": "episode finished"})

        if action not in ACTIONS:
            raise ValueError(f"Ação inválida: {action}. Use 0:UP,1:RIGHT,2:DOWN,3:LEFT")

        dr, dc = ACTIONS[action]
        r, c = self._state
        cand: Pos = (r + dr, c + dc)

        reward = self.step_cost
        info: Dict = {"action_name": ACTION_NAMES[action]}

        if not self._inside(cand) or cand in self.walls:
            # Movimento bloqueado
            reward += self.wall_penalty
            next_state = self._state
            done = False
            info["blocked"] = True
        else:
            # Movimento válido
            next_state = cand
            cell = self._cell(next_state)
            if next_state in self.traps or cell == 'T':
                reward += self.trap_penalty
                done = False
                info["trap"] = True
            elif next_state == self.goal or cell == 'G':
                reward += self.goal_reward
                done = True
                info["goal"] = True
            else:
                done = False

        self._state = next_state
        self._steps += 1

        if self._steps >= self.max_steps and not done:
            done = True
            info["max_steps"] = True

        self._done = done
        return StepResult(self._state, reward, done, info)

    def render(self, path: Optional[List[Pos]] = None) -> None:
        """
        Render ASCII.
        - 'A' = agente
        - '*' = caminho percorrido (se path fornecido)
        """
        marks = {p: '*' for p in (path or [])}
        for i in range(self.n_rows):
            line = []
            for j in range(self.n_cols):
                p = (i, j)
                if p == self._state:
                    ch = 'A'
                else:
                    ch = self.grid_raw[i][j]
                    if p != self.start and p != self.goal and p not in self.walls and p not in self.traps:
                        if p in marks:
                            ch = '*'
                line.append(ch)
            print(''.join(line))
        print()

    def legal_actions(self, pos: Optional[Pos] = None) -> List[int]:
        pos = self._state if pos is None else pos
        legal = []
        for a, (dr, dc) in ACTIONS.items():
            r, c = pos
            nxt = (r + dr, c + dc)
            if self._inside(nxt) and nxt not in self.walls:
                legal.append(a)
        return legal


# ---------------- Base do Agente (para os alunos implementarem) ----------------

class Agent(ABC):
    """
    Interface do Agente: os alunos DEVEM implementar pelo menos:
      - select_action(state): escolher uma ação
      - observe(s, a, r, s_next, done): (opcional) atualizar a política/valores
      - train_episode(env): (opcional) loop de treino por episódio
    """

    @abstractmethod
    def select_action(self, state: Pos) -> int:
        """Retorna uma ação (0,1,2,3) a partir do estado (linha,coluna)."""
        raise NotImplementedError

    def observe(self, s: Pos, a: int, r: float, s_next: Pos, done: bool) -> None:
        """Atualize sua estratégia (Q-Table, política, etc.) se desejar."""
        pass

    def train_episode(self, env: MazeEnv, max_steps: int = 500) -> float:
        """
        Exemplo de loop de treino (pode ser sobrescrito).
        Retorna a recompensa total acumulada no episódio.
        """
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = self.select_action(s)
            step = env.step(a)
            self.observe(s, a, step.reward, step.next_state, step.done)
            total += step.reward
            s = step.next_state
            if step.done:
                break
        return total


# ----------------- Exemplo de grade/execução (sem agente implementado) -----------------

DEFAULT_GRID = [
    # 0123456789 (colunas)
    "###########",  # 0
    "#S..T....G#",  # 1
    "#.#.###.#.#",  # 2
    "#.#...#.#.#",  # 3
    "#.###.#.#.#",  # 4
    "#.....#...#",  # 5
    "#####.###.#",  # 6
    "#...T.....#",  # 7
    "###########",  # 8
]
# Interpretação:
# - Paredes (#) cercam o labirinto e criam corredores
# - S (start) em (1,1), G (goal) em (1,9), T (traps) em (1,4) e (7,4)
# - '.' livres

# ----------------- Exemplo de "esqueleto" de um agente para os alunos -----------------

class MyAgent(Agent):
    """
    TODO pelos alunos:
    - implementar uma estratégia: por exemplo, BFS/DFS determinístico, Dijkstra, A*, ou Q-Learning com política greedy.
    - garantir que select_action retorne uma ação válida (0..3).
    - (opcional) manter memória de visitados para evitar loops.
    """
    def __init__(self, env: MazeEnv):
        self.env = env
        # TODO: inicializar estruturas (ex.: fila/pilha, heurística, Q-table, etc.)

    def select_action(self, state: Pos) -> int:
        # TODO: escolher ação determinística que avance até a saída
        # Sugestão: implemente uma busca (DFS/BFS) offline para gerar um caminho
        # e depois, em tempo de execução, siga o caminho com as ações mapeadas.
        raise NotImplementedError("Implemente sua política aqui.")

    def observe(self, s: Pos, a: int, r: float, s_next: Pos, done: bool) -> None:
        # TODO (opcional): atualizar conhecimento/valores
        pass

    # (opcional) sobrescrever train_episode para incluir lógica própria
    # def train_episode(self, env: MazeEnv, max_steps: int = 500) -> float:
    #     return super().train_episode(env, max_steps)


# ----------------- Execução de referência -----------------

def run_greedy_demo(env: MazeEnv, agent: Agent, max_steps: int = 200, show: bool = True) -> float:
    """
    Runner padrão: executa um episódio, renderiza e retorna a recompensa total.
    OBS: Com MyAgent sem implementar, isto levantará NotImplementedError.
    """
    path = []
    s = env.reset()
    total = 0.0
    for t in range(max_steps):
        a = agent.select_action(s)  # <- deve estar implementado pelos alunos
        result = env.step(a)
        path.append(result.next_state)
        total += result.reward
        if show:
            env.render(path)
        if result.done:
            break
        s = result.next_state
    return total


if __name__ == "__main__":
    env = MazeEnv(DEFAULT_GRID,
                  step_cost=-1.0,
                  wall_penalty=-2.0,
                  trap_penalty=-5.0,
                  goal_reward=50.0,
                  max_steps=300)

    # Substitua MyAgent por sua implementação.
    agent = BFSAgent(env, avoid_traps=True)

    # Isto vai falhar (de propósito) até os alunos implementarem select_action.
    try:
        score = run_greedy_demo(env, agent, show=True)
        print(f"Recompensa total: {score}")
    except NotImplementedError as e:
        print("Implementação pendente do agente:", e)
