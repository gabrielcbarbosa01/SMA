"""
Simulador de Rede de Filas - G/G/c/K com topologia generica
Disciplina: Simulacao e Metodos Analiticos
Estudantes: Gustavo Melleu e Gabriel Barbosa

Uso:
    python3 simulador_rede_filas.py --config config_m6_tandem.json

Formato do config.json:
{
  "max_rand": 100000,
  "primeiro_chegada": 1.5,
  "filas": [
    {
      "id": 1,
      "servidores": 2,
      "capacidade": 3,
      "chegada_min": 1,
      "chegada_max": 4,
      "atendimento_min": 3,
      "atendimento_max": 4
    },
    {
      "id": 2,
      "servidores": 1,
      "capacidade": 5,
      "chegada_min": null,
      "chegada_max": null,
      "atendimento_min": 2,
      "atendimento_max": 3
    }
  ],
  "roteamento": [
    {"de": 1, "para": 2, "prob": 1.0},
    {"de": 2, "para": null, "prob": 1.0}
  ]
}

Observacoes:
- chegada_min/chegada_max = null significa que a fila nao recebe clientes externos
- roteamento "para": null significa saida do sistema
- as probabilidades de roteamento de uma mesma fila devem somar 1.0
"""

import random
import heapq
import json
import sys
import argparse


# ────────────────────────────────────────────────────────────────────────────
# Gerador de numeros aleatorios com contador global
# ────────────────────────────────────────────────────────────────────────────

class Gerador:
    def __init__(self, seed=42):
        random.seed(seed)
        self.usado = 0
        self.esgotado = False

    def rand(self):
        if self.esgotado:
            return None
        self.usado += 1
        return random.random()

    def uniforme(self, a, b):
        """Gera valor uniforme entre a e b usando 1 numero aleatorio."""
        r = self.rand()
        if r is None:
            return None
        return a + r * (b - a)


# ────────────────────────────────────────────────────────────────────────────
# Estrutura de Fila
# ────────────────────────────────────────────────────────────────────────────

class Fila:
    def __init__(self, fila_id, servidores, capacidade, at_min, at_max):
        self.id = fila_id
        self.servidores = servidores
        self.capacidade = capacidade   # capacidade total = fila + servidores
        self.at_min = at_min
        self.at_max = at_max

        # Estado
        self.clientes = 0              # clientes atualmente no sistema desta fila
        self.servidores_livres = [0.0] * servidores  # tempo em que cada servidor fica livre
        self.perdas = 0

        # Acumuladores para distribuicao de probabilidades
        self.tempo_por_estado = [0.0] * (capacidade + 1)
        self.ultimo_evento_t = 0.0

    def registrar_estado(self, t_atual):
        """Atualiza tempo acumulado do estado atual ate t_atual."""
        delta = t_atual - self.ultimo_evento_t
        if delta > 0 and self.clientes <= self.capacidade:
            self.tempo_por_estado[self.clientes] += delta
        self.ultimo_evento_t = t_atual

    def servidor_disponivel(self, t):
        """Retorna o servidor mais cedo livre (ou None se todos ocupados e fila cheia)."""
        # Conta servidores ocupados no momento t
        ocupados = sum(1 for s in self.servidores_livres if s > t)
        # Se ha servidor livre E ha espaco
        if ocupados < self.servidores:
            return True  # pode atender imediatamente
        return False

    def proximo_servidor(self, t):
        """Retorna indice do servidor que ficara livre mais cedo."""
        return min(range(self.servidores), key=lambda i: self.servidores_livres[i])


# ────────────────────────────────────────────────────────────────────────────
# Evento
# ────────────────────────────────────────────────────────────────────────────

CHEGADA = 'chegada'
SAIDA   = 'saida'

class Evento:
    def __init__(self, t, tipo, fila_id, cliente_id=None):
        self.t = t
        self.tipo = tipo
        self.fila_id = fila_id
        self.cliente_id = cliente_id

    def __lt__(self, other):
        return self.t < other.t


# ────────────────────────────────────────────────────────────────────────────
# Simulador
# ────────────────────────────────────────────────────────────────────────────

class SimuladorRedeFilas:
    def __init__(self, config):
        self.max_rand = config.get('max_rand', 100000)
        self.primeiro_chegada = config.get('primeiro_chegada', 1.5)
        self.gen = Gerador(seed=42)

        # Construir filas
        self.filas = {}
        for f in config['filas']:
            fila = Fila(
                fila_id=f['id'],
                servidores=f['servidores'],
                capacidade=f['capacidade'],
                at_min=f['atendimento_min'],
                at_max=f['atendimento_max']
            )
            self.filas[f['id']] = fila

            # Parametros de chegada externa (pode ser null)
            fila.ch_min = f.get('chegada_min')
            fila.ch_max = f.get('chegada_max')

        # Roteamento: dict fila_id -> lista de (prob_acumulada, destino_id_ou_None)
        self.roteamento = {}
        for r in config.get('roteamento', []):
            origem = r['de']
            if origem not in self.roteamento:
                self.roteamento[origem] = []
            self.roteamento[origem].append((r['prob'], r['para']))

        # Converter probabilidades em acumuladas
        for fila_id, rotas in self.roteamento.items():
            acum = 0.0
            novas = []
            for prob, dest in rotas:
                acum += prob
                novas.append((acum, dest))
            self.roteamento[fila_id] = novas

        # Fila de eventos (heap)
        self.heap = []
        self.t = 0.0
        self.cliente_counter = 0

    def _novo_cliente(self):
        self.cliente_counter += 1
        return self.cliente_counter

    def _agendar(self, evento):
        heapq.heappush(self.heap, evento)

    def _tempo_uniforme(self, a, b):
        if self.gen.usado >= self.max_rand:
            return None
        return self.gen.uniforme(a, b)

    def _escolher_destino(self, fila_id):
        """Escolhe proximo destino baseado nas probabilidades de roteamento.
        So consome um numero aleatorio se houver mais de um destino possivel."""
        rotas = self.roteamento.get(fila_id, [])
        if not rotas:
            return None
        # Roteamento deterministico (unico destino com prob=1.0) nao consome rand
        if len(rotas) == 1:
            return rotas[0][1]
        r = self.gen.rand()
        if r is None:
            return 'STOP'
        for prob_acum, dest in rotas:
            if r <= prob_acum:
                return dest
        return rotas[-1][1]  # fallback

    def _processar_chegada(self, evento):
        fila = self.filas[evento.fila_id]
        t = evento.t

        fila.registrar_estado(t)

        if fila.clientes < fila.capacidade:
            fila.clientes += 1

            # Agendar saida
            at = self._tempo_uniforme(fila.at_min, fila.at_max)
            if at is None:
                return False  # aleatorios esgotados

            # Encontrar servidor que ficara livre mais cedo
            idx = fila.proximo_servidor(t)
            inicio_atend = max(t, fila.servidores_livres[idx])
            fim_atend = inicio_atend + at
            fila.servidores_livres[idx] = fim_atend

            saida = Evento(fim_atend, SAIDA, fila.id, evento.cliente_id)
            self._agendar(saida)
        else:
            fila.perdas += 1

        # Agendar proxima chegada externa (se esta fila tem chegadas externas)
        if fila.ch_min is not None and fila.ch_max is not None:
            intervalo = self._tempo_uniforme(fila.ch_min, fila.ch_max)
            if intervalo is None:
                return False
            proxima = Evento(t + intervalo, CHEGADA, fila.id, self._novo_cliente())
            self._agendar(proxima)

        return True

    def _processar_saida(self, evento):
        fila = self.filas[evento.fila_id]
        t = evento.t

        fila.registrar_estado(t)
        fila.clientes -= 1

        # Rotear cliente para proxima fila (ou saida do sistema)
        destino = self._escolher_destino(fila.id)
        if destino == 'STOP':
            return False
        if destino is not None:
            chegada = Evento(t, CHEGADA, destino, evento.cliente_id)
            self._agendar(chegada)

        return True

    def simular(self):
        # Agendar chegadas iniciais para filas com chegadas externas
        for fila_id, fila in self.filas.items():
            if fila.ch_min is not None:
                # Primeiro cliente chega no tempo configurado
                intervalo = self._tempo_uniforme(fila.ch_min, fila.ch_max)
                if intervalo is None:
                    break
                # Usar o primeiro_chegada como tempo do primeiro evento
                primeiro_t = self.primeiro_chegada
                chegada = Evento(primeiro_t, CHEGADA, fila_id, self._novo_cliente())
                self._agendar(chegada)

        while self.heap:
            if self.gen.usado >= self.max_rand:
                break

            evento = heapq.heappop(self.heap)
            self.t = evento.t

            if evento.tipo == CHEGADA:
                ok = self._processar_chegada(evento)
                if not ok:
                    break
            elif evento.tipo == SAIDA:
                ok = self._processar_saida(evento)
                if not ok:
                    break

        # Registrar estado final para todas as filas
        for fila in self.filas.values():
            fila.registrar_estado(self.t)

        return self.t

    def relatorio(self):
        print("=" * 70)
        print(f"SIMULACAO DE REDE DE FILAS")
        print(f"Tempo global da simulacao: {self.t:.4f}")
        print(f"Numeros aleatorios utilizados: {self.gen.usado}")
        print("=" * 70)

        for fila_id, fila in sorted(self.filas.items()):
            print(f"\n--- Fila {fila_id} (G/G/{fila.servidores}/{fila.capacidade}) ---")
            print(f"Numero de perdas: {fila.perdas}")
            tempo_total = sum(fila.tempo_por_estado)
            print(f"Distribuicao de probabilidades por estado:")
            for estado, tempo in enumerate(fila.tempo_por_estado):
                prob = tempo / tempo_total if tempo_total > 0 else 0
                label = ""
                if estado == fila.capacidade:
                    label = " [sistema cheio - perdas]"
                elif estado == fila.capacidade - fila.servidores + 1 if fila.servidores > 1 else estado == fila.capacidade:
                    pass
                print(f"  Estado {estado}: tempo={tempo:.4f}  prob={prob:.6f} ({prob*100:.4f}%){label}")

        print("=" * 70)


# ────────────────────────────────────────────────────────────────────────────
# Configuracao default (especificacao do trabalho M6)
# ────────────────────────────────────────────────────────────────────────────

CONFIG_DEFAULT = {
    "max_rand": 100000,
    "primeiro_chegada": 1.5,
    "filas": [
        {
            "id": 1,
            "servidores": 2,
            "capacidade": 3,
            "chegada_min": 1,
            "chegada_max": 4,
            "atendimento_min": 3,
            "atendimento_max": 4
        },
        {
            "id": 2,
            "servidores": 1,
            "capacidade": 5,
            "chegada_min": None,
            "chegada_max": None,
            "atendimento_min": 2,
            "atendimento_max": 3
        }
    ],
    "roteamento": [
        {"de": 1, "para": 2, "prob": 1.0},
        {"de": 2, "para": None, "prob": 1.0}
    ]
}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulador de Rede de Filas G/G/c/K')
    parser.add_argument('--config', type=str, default=None,
                        help='Caminho para arquivo JSON de configuracao')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print("Usando configuracao default (trabalho M6: Filas em Tandem)")
        config = CONFIG_DEFAULT

    sim = SimuladorRedeFilas(config)
    sim.simular()
    sim.relatorio()