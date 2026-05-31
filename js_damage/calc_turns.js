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

function calcSingle(item) {
  const gen = item.gen || 9;
  const attacker = buildPokemon(gen, item.attacker);
  const defender = buildPokemon(gen, item.defender);
  const move = new Move(gen, item.move.name, item.move.options || {});

  let field = undefined;
  if (item.field) {
    try {
      field = new Field(item.field);
    } catch (e) {
      // ignore malformed field, fall back to default
    }
  }

  const result = calculate(gen, attacker, defender, move, field);
  const damage = result.damage;
  const min = Array.isArray(damage) ? Math.min(...damage) : damage;
  const max = Array.isArray(damage) ? Math.max(...damage) : damage;

  const hpFraction = item.defender.hp_fraction !== undefined ? item.defender.hp_fraction : 1.0;
  const defenderHP = Math.max(1, Math.round(defender.stats.hp * hpFraction));

  const avg = Array.isArray(damage) ? damage.reduce((a,b)=>a+b,0)/damage.length : damage;

  const turns_min = Math.ceil(defenderHP / max);
  const turns_max = Math.ceil(defenderHP / min);
  const turns_avg = Math.ceil(defenderHP / avg);

  const defenderMaxHP = defender.stats.hp || 1;
  return {
    turns_avg,
    turns_min,
    turns_max,
    min_pct: (min / defenderMaxHP) * 100,
    max_pct: (max / defenderMaxHP) * 100,
    avg_pct: (avg / defenderMaxHP) * 100,
    defender_max_hp: defenderMaxHP,
  };
}

// Batch mode: if input.batch is an array, process each item and return array of results
if (Array.isArray(input.batch)) {
  const results = input.batch.map(item => {
    try {
      return calcSingle(item);
    } catch (e) {
      return {error: String(e.message || e)};
    }
  });
  process.stdout.write(JSON.stringify(results));
} else {
  // Single-item mode (existing behavior — unchanged)
  try {
    process.stdout.write(JSON.stringify(calcSingle(input)));
  } catch (e) {
    process.stderr.write(String(e.stack || e));
    process.exit(1);
  }
}
