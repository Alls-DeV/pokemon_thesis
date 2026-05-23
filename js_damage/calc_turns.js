#!/usr/bin/env node
import {calculate, Pokemon, Move, Field} from '@smogon/calc';
import {Generations} from '@smogon/calc';

// Read stdin JSON
const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}
const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));

function buildPokemon(gen, data) {
  return new Pokemon(gen, data.species, {
    level: data.level || 100,
    item: data.item || undefined,
    ability: data.ability || undefined,
    nature: data.nature || undefined,
    evs: data.evs || undefined,
    ivs: data.ivs || undefined,
    boosts: data.boosts || undefined,
    status: data.status || undefined,
  });
}

try {
  const gen = input.gen || 9;
  const attacker = buildPokemon(gen, input.attacker);
  const defender = buildPokemon(gen, input.defender);
  const move = new Move(gen, input.move.name, input.move.options || {});

  let field = undefined;
  if (input.field) {
    try {
      field = new Field(input.field);
    } catch (e) {
      // ignore malformed field, fall back to default
    }
  }

  const result = calculate(gen, attacker, defender, move, field);
  const damage = result.damage;
  const min = Array.isArray(damage) ? Math.min(...damage) : damage;
  const max = Array.isArray(damage) ? Math.max(...damage) : damage;
  
  // Calculate true absolute defender HP based on their fraction and calculated max HP
  const hpFraction = input.defender.hp_fraction !== undefined ? input.defender.hp_fraction : 1.0;
  const defenderHP = Math.max(1, Math.round(defender.stats.hp * hpFraction));

  // naive expected hits (average roll)
  const avg = Array.isArray(damage) ? damage.reduce((a,b)=>a+b,0)/damage.length : damage;

  // compute worst / best case turns to KO
  const turns_min = Math.ceil(defenderHP / max); // best case (high roll each time)
  const turns_max = Math.ceil(defenderHP / min); // worst case (low roll each time)
  const turns_avg = Math.ceil(defenderHP / avg);

  // Output only the average number of turns to KO as requested
  process.stdout.write(JSON.stringify({turns_avg}));
} catch (e) {
  process.stderr.write(String(e.stack || e));
  process.exit(1);
}
