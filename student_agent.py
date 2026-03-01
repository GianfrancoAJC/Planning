import re

class AssemblyAgent:
    def __init__(self):
        self.system_prompt = (
            "Eres un planificador experto en lógica de predicados. Tu misión es alcanzar la meta siguiendo estas reglas estrictas:\n\n"
            "1. REGLA DE ARMONÍA: Solo puedes usar (attack) si hay 'Harmony'.\n"
            "2. RESTAURACIÓN: Si al inicio el STATEMENT dice que 'x craves y', la 'Harmony' está rota.\n"
            "   - Para recuperar 'Harmony', DEBES ejecutar primero: (feast x y) y luego (succumb x).\n"
            "3. ACCIONES PAREADAS:\n"
            "   - Para ELIMINAR un deseo: (feast x y) -> (succumb x)\n"
            "   - Para CREAR un deseo: (attack x) -> (overcome x z)\n"
            "4. LONGITUD: El plan debe ser de 2, 4 o 6 acciones. Si hay deseos iniciales que no están en la meta, DEBES eliminarlos primero.\n"
            "5. FORMATO: Solo acciones (accion arg1) o (accion arg1 arg2). Sin la palabra 'object'.\n"
        )

        self.few_shot_examples = '''
[STATEMENT]
Initial: object a craves object b, harmony, planet object c, province object a. Goal: object b craves object a.
Explicación: Hay un deseo (a->b) que no está en la meta. Debo limpiarlo primero para recuperar armonía.
[PLAN]
(feast a b)
(succumb a)
(attack b)
(overcome b a)

[STATEMENT]
Initial: harmony, planet object a, planet object b, province object a, province object b. Goal: object a craves object c and object b craves object a.
Explicación: Inicio limpio. Solo construyo.
[PLAN]
(attack a)
(overcome a c)
(attack b)
(overcome b a)
'''

    def _parse_actions(self, text):
        patron = re.compile(r"\((attack|overcome|feast|succumb)\s+([a-z])(?:\s+([a-z]))?\)")
        acciones = []
        for match in patron.finditer(text.lower()):
            verbo, arg1, arg2 = match.groups()
            acciones.append(f"({verbo} {arg1} {arg2})" if arg2 else f"({verbo} {arg1})")
        
        # Deduplicación y ajuste de longitud
        res = []
        for a in acciones:
            if not res or a != res[-1]: res.append(a)
        
        n = len(res)
        if n > 6: res = res[:6]
        elif n in [3, 5]: res = res[:-1] # Mantener siempre pares
        return res

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        # Forzamos al modelo a detectar los 'craves' iniciales antes de responder
        prompt = (
            f"{self.few_shot_examples}\n\n"
            f"[STATEMENT]\n{scenario_context}\n\n"
            "Detecta si hay deseos iniciales. Si los hay, límpialos con feast y succumb. Luego construye la meta.\n"
            "[PLAN]\n"
        )

        respuesta = llm_engine_func(
            prompt=prompt,
            system=self.system_prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=200
        )
        return self._parse_actions(respuesta)
