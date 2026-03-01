import re


class AssemblyAgent:
    def __init__(self):
        self.system_prompt = (
            "Eres un planificador simbólico experto en planificación automática.\n"
            "Dominio permitido:\n"
            "Acciones válidas: attack, overcome, feast, succumb.\n"
            "Reglas estrictas del dominio:\n"
            "- Nunca uses la palabra 'object'.\n"
            "- Nunca uses la palabra 'from'.\n"
            "- Los argumentos deben ser solo símbolos simples como a, b, c.\n"
            "- Formato exacto de acción con dos argumentos: (accion a b)\n"
            "- Formato exacto con un argumento: (accion a)\n"
            "- No escribas explicaciones.\n"
            "- No agregues texto adicional.\n"
            "- Solo imprime acciones válidas en líneas separadas.\n"
            "- El plan debe ser mínimo.\n"
            "- La última acción debe lograr exactamente el objetivo.\n"
        )

        self.few_shot_examples = '''
Ejemplo:

[STATEMENT]
As initial conditions I have that harmony, planet a, planet b, province a and province b.
My goal is to have that a craves b.

[PLAN]
(attack a)
(overcome a b)

Ejemplo:

[STATEMENT]
As initial conditions I have that a craves c, harmony, planet b and province a.
My goal is to have that c craves a.

[PLAN]
(feast a c)
(succumb a)
(attack c)
(overcome c a)
'''

    def _parse_actions(self, text):
        lines = text.split("\n")
        acciones = []

        patron = re.compile(r"^\((attack|overcome|feast|succumb)\s+[a-z](?:\s+[a-z])?\)$")

        for line in lines:
            line = line.strip()
            if patron.match(line):
                acciones.append(line)

        return acciones

    def solve(self, scenario_context: str, llm_engine_func) -> list:

        prompt = f"""
{self.few_shot_examples}

Resuelve el siguiente problema.

Reglas:
- Usa únicamente las acciones permitidas.
- No uses palabras adicionales.
- El plan debe ser mínimo.
- Solo incluye acciones necesarias.
- La última acción debe satisfacer exactamente la meta.

[STATEMENT]
{scenario_context}

[PLAN]
"""

        respuesta = llm_engine_func(
            prompt=prompt,
            system=self.system_prompt,
            do_sample=False,
            max_new_tokens=150
        )

        acciones = self._parse_actions(respuesta)

        return acciones
