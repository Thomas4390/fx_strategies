# 08 — File I/O & Global Variables

Lire des fichiers (ex. `macro_cache.csv`) et persister des états (peak equity, DD
flag) entre redémarrages.

## File I/O

### Localisation des fichiers

MQL5 a un sandbox strict : un programme ne peut écrire que dans deux dossiers :

| Dossier | Résolution physique | Flag |
|---------|---------------------|------|
| `MQL5/Files/` (par-terminal) | `<TerminalDataPath>/MQL5/Files/` | (défaut) |
| `Common/Files/` (partagé) | `<CommonDataPath>/Files/` | `FILE_COMMON` |

`<TerminalDataPath>` : voir `Fichier → Ouvrir le dossier de données` dans le terminal.
`<CommonDataPath>` : `%APPDATA%/MetaQuotes/Terminal/Common/Files` (Windows) ou
`~/.wine/drive_c/users/<u>/AppData/Roaming/MetaQuotes/Terminal/Common/Files` (Linux/Wine).

**Choix recommandé** pour `macro_cache.csv` : `Common/Files/` (`FILE_COMMON`) — un seul
fichier partagé entre tous les terminaux MT5 du système.

### Lecture CSV

```mql5
int h = FileOpen("macro_cache.csv",
                 FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON,
                 ',');
if(h == INVALID_HANDLE)
{
    PrintFormat("Cannot open macro_cache.csv: %d", GetLastError());
    return false;
}

// Skip header (5 colonnes)
for(int i = 0; i < 5; i++) FileReadString(h);

// Lecture ligne 1
string ts          = FileReadString(h);
double spread      = StringToDouble(FileReadString(h));
int    unemp_rising = (int)StringToInteger(FileReadString(h));
double threshold   = StringToDouble(FileReadString(h));
int    macro_ok    = (int)StringToInteger(FileReadString(h));

FileClose(h);
```

### Flags `FileOpen`

| Flag | Sens |
|------|------|
| `FILE_READ` | Lecture |
| `FILE_WRITE` | Écriture (truncate) |
| `FILE_BIN` | Binaire (sinon texte) |
| `FILE_TXT` | Texte ligne par ligne |
| `FILE_CSV` | Texte CSV avec délimiteur (3e arg de `FileOpen`) |
| `FILE_ANSI` | ANSI (sinon UNICODE-16) |
| `FILE_UNICODE` | UNICODE-16 LE |
| `FILE_COMMON` | Dossier partagé `<CommonDataPath>/Files/` |
| `FILE_SHARE_READ`/`WRITE` | Partage avec autres processus |

### Écriture (rare en EA, plus pour logging)

```mql5
int h = FileOpen("fx_log.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
FileWrite(h, "timestamp", "sleeve", "action", "symbol", "lots", "price");
FileWrite(h, TimeToString(TimeGMT()), "MR_Macro", "BUY", "EURUSD", 0.10, 1.0850);
FileClose(h);
```

`FileWrite` ajoute automatiquement le séparateur entre arguments et le `\n` en fin
de ligne.

### Append (mode ajout)

```mql5
int h = FileOpen("fx_log.csv", FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
FileSeek(h, 0, SEEK_END);
FileWrite(h, ...);
FileClose(h);
```

## Global Variables — persistance entre redémarrages

Différentes des `extern` C : ce sont des **variables persistées sur disque** par le
terminal, accessibles entre redémarrages (et entre EAs si on partage le nom).

### Lire / écrire

```mql5
GlobalVariableSet("FX_PEAK_EQUITY", 10500.0);    // écrit
double peak = GlobalVariableGet("FX_PEAK_EQUITY"); // lit (0.0 si inexistante)

if(GlobalVariableCheck("FX_PEAK_EQUITY"))
    peak = GlobalVariableGet("FX_PEAK_EQUITY");
else
    GlobalVariableSet("FX_PEAK_EQUITY", AccountInfoDouble(ACCOUNT_EQUITY));
```

### Suppression

```mql5
GlobalVariableDel("FX_DD_TRIGGERED");
```

### Itération

```mql5
int total = GlobalVariablesTotal();
for(int i = 0; i < total; i++)
{
    string name = GlobalVariableName(i);
    double val  = GlobalVariableGet(name);
}
```

### Cas d'usage dans ce projet

| Variable | Sens |
|----------|------|
| `FX_PEAK_EQUITY` | High-water mark equity, mis à jour à chaque tick |
| `FX_DD_TRIGGERED` | 0/1 — flag circuit-breaker (verrou jusqu'à reset manuel) |
| `FX_GLOBAL_LEVERAGE` | Levier global recalculé chaque jour à 21:00 UTC |
| `FX_LAST_DAILY_RECOMP` | Timestamp du dernier recompute daily (anti-double-trigger) |

## Différence Global Variables vs `static` vs `input`

| | Global Variable | `static` local | `input` |
|--|-----------------|----------------|---------|
| Survit à un redémarrage | ✅ | ❌ | ✅ (par config terminal) |
| Survit à un changement d'inputs / recompil | ✅ | ❌ | ✅ |
| Survit à un changement de chart | ✅ | ❌ | ✅ |
| Modifiable par l'utilisateur | via panneau Outils → Variables globales | ❌ | ✅ via panneau EA |
| Accessible par d'autres EAs | ✅ | ❌ | ❌ |

**Règle** : pour persister un état qui doit survivre aux redémarrages →
`GlobalVariable`. Pour un état session-only → `static` ou variable membre de classe.

## Pattern : init avec valeur par défaut

```mql5
double GetOrInit(string name, double default_value)
{
    if(!GlobalVariableCheck(name))
        GlobalVariableSet(name, default_value);
    return GlobalVariableGet(name);
}

void OnInit()
{
    double peak = GetOrInit("FX_PEAK_EQUITY", AccountInfoDouble(ACCOUNT_EQUITY));
}
```

## Reset manuel via input

```mql5
input bool Inp_ResetDDState = false;

int OnInit()
{
    if(Inp_ResetDDState)
    {
        GlobalVariableSet("FX_PEAK_EQUITY", AccountInfoDouble(ACCOUNT_EQUITY));
        GlobalVariableDel("FX_DD_TRIGGERED");
        Print("DD state reset");
    }
    return INIT_SUCCEEDED;
}
```

L'utilisateur passe `true`, recharge l'EA, le reset est consommé. Pour empêcher de
re-reset à la prochaine recompil, on peut faire :

```mql5
if(Inp_ResetDDState && !GlobalVariableCheck("FX_RESET_CONSUMED"))
{
    // ...reset...
    GlobalVariableSet("FX_RESET_CONSUMED", 1.0);
}
```

## Voir aussi

- [03_trade_operations.md](./03_trade_operations.md) — `AccountInfoDouble`
- [10_pitfalls.md](./10_pitfalls.md) — sandbox MQL5/Files
