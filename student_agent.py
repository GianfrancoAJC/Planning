import re

class AssemblyAgent:
    def __init__(self):
        self.system_prompt = (
            "Eres un planificador simbólico experto en planificación automática.\n"
            "Debes generar únicamente una secuencia de acciones válidas.\n"
            "No expliques nada.\n"
            "No agregues texto adicional.\n"
            "Cada acción debe estar en formato exacto:\n"
            "(accion argumento1 argumento2)\n"
            "Si la acción tiene un solo argumento:\n"
            "(accion argumento)\n"
            "Salida estrictamente en líneas separadas."
        )

        # Few-shot mínimo para mantener tiempo bajo
        self.few_shot_examples = """
Ejemplo:

[STATEMENT]
As initial conditions I have that, harmony, planet object a, planet object b, province object a and province object b.
My goal is to have that object a craves object b.

[PLAN]
(attack a)
(overcome a b)

Ejemplo:

[STATEMENT]
As initial conditions I have that, object a craves object c, harmony, planet object b and province object a.
My goal is to have that object c craves object a.

[PLAN]
(feast a c)
(succumb a)
(attack c)
(overcome c a)
"""

    def _parse_actions(self, text):
        lines = text.split("\n")
        acciones = []

        for line in lines:
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                acciones.append(line)

        return acciones

    def solve(self, scenario_context: str, llm_engine_func) -> list:

        prompt = f"""
{self.few_shot_examples}

Ahora resuelve el siguiente problema.

[STATEMENT]
{scenario_context}

[PLAN]
"""

        respuesta = llm_engine_func(
            prompt=prompt,
            system=self.system_prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=256
        )

        acciones = self._parse_actions(respuesta)

        return acciones