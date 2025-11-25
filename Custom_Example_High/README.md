# Custom Example

This directory serves as an example of how to structure a custom mapper for Crusader Conflicts.

## File Structure and Purpose

- `Cultures/`: This directory contains XML files that define the cultures for your custom map.
    - `OfficialCC_Custom_Default_Cultures.xml`: An example culture definition file.
- `Factions/`: This directory contains XML files that define the factions and their units.
    - `OfficialCC_Custom_Default_Factions.xml`: An example faction definition file.
- `Terrains/`: This directory contains XML files that define the terrains for your custom map.
- `Titles/` (Optional): This directory can contain XML files to assign specific men-at-arms units to landed titles (e.g., counties, duchies, kingdoms). This is an optional feature.
    - `Counties.xml`: Example for county-level titles.
    - `Duchies.xml`: Example for duchy-level titles.
    - `Kingdoms.xml`: Example for kingdom-level titles.
- `Mods.xml`: This file lists the mods that this custom mapper depends on.
- `TimePeriod.xml`: This file defines the time period for your custom map.
- `background.png`: A background image for the custom map in the launcher.

## Required Files

For a custom mapper to be valid, the following files are required:

- `Mods.xml`
- `TimePeriod.xml`
- At least one XML file in the `Cultures/` directory.
- At least one XML file in the `Factions/` directory.
- At least one XML file in the `Terrains/` directory.

Submods, submod addons, and other general addons are not required.

## Loading Order

The files within the `Cultures`, `Factions`, `Terrains`, and `Titles` directories are loaded alphabetically. If you have multiple files in these directories, their contents will be merged.

The general loading order for the mapper is as follows:
1.  **Cultures**: Defines the cultures.
2.  **Factions**: Defines factions and their units.
3.  **Terrains**: Defines the terrain mapping.
4.  **Titles** (Optional): Assigns specific units to landed titles, overriding culture-based units.
