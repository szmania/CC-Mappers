# CC-Mappers
Mappers for [Crusader Conflicts](https://github.com/szmania/Crusader-Wars/releases/latest), Community Successor for Crusader Wars mod.

## Mapper Loading Order

The Crusader Conflicts mod loads culture and faction mapping XML files in a specific order to allow for overrides. The loading process is as follows:

1.  **Base Mappers**: The official playthrough base mappers are loaded first. These files are identified by the naming convention `OfficialCC_<mapper_tag>_*.xml`.
2.  **Submod Mappers**: After the base mappers, all other XML files in the cultures and factions directories are loaded in alphabetical order. These are typically submod mappers.

This loading order means that any mapping defined in a submod mapper will overwrite a mapping for the same culture or faction from a base mapper. If multiple submod mappers define the same mapping, the one loaded last (alphabetically) will take precedence.


## Custom Mappers

### Custom Example

This directory serves as an example of how to structure a custom mapper for Crusader Conflicts.

### File Structure and Purpose

- `Cultures/`: This directory contains XML files that define what Crusader Kings 3 heritages and cultures are mapped to what factions in Total War: Attila.
    - `OfficialCC_Custom_Default_Cultures.xml`: An example culture definition file.
- `Factions/`: This directory contains XML files that define the factions and their units.
    - `OfficialCC_Custom_Default_Factions.xml`: An example faction definition file.
- `Terrains/`: This directory contains XML files that define the terrains for your custom map, including coastal regions, straits, river crossings, and settlement coordinates.
- `Titles/` (Optional): This directory can contain XML files to assign specific men-at-arms units to landed titles (e.g., counties, duchies, kingdoms). This is an optional feature.
    - `Counties.xml`: Example for county-level titles.
    - `Duchies.xml`: Example for duchy-level titles.
    - `Kingdoms.xml`: Example for kingdom-level titles.
- `Mods.xml`: This file lists the mods that this custom mapper depends on, including optional submods. Note the order in which the Submod appears is the load order the submod will load. Pack files at the top of the file will load last.
- `TimePeriod.xml`: This file defines the time period for your custom map.
- `tag.txt`: The value in this file is the name that shows in the dropdown within the Crusader Conflicts Custom mapper tab. This is the tag for your custom mapper, and this value MUST start with "Custom" in order to appear in the Custom mapper screen. You can have multiple folders with the same tag.txt value, if you want, for example, to have different units for different TimePeriod.xml values. eg. Early, High, Late, Renaissance periods.
- `background.png`: A background image for the custom map in the launcher and Total War: Attila.

### Required Files

For a custom mapper to be valid, the following files are required:

- `Mods.xml`
- `TimePeriod.xml`
- `tag.txt`
- At least one XML file in the `Cultures/` directory.
- At least one XML file in the `Factions/` directory.
- At least one XML file in the `Terrains/` directory.

Submods, submod addons, and other general addons are not required.

### Loading Order

The files within the `Cultures`, `Factions`, `Terrains`, and `Titles` directories are loaded alphabetically. If you have multiple files in these directories, their contents will be merged.

The general loading order for the mapper is as follows:
1.  **Cultures**: Defines the cultures.
2.  **Factions**: Defines factions and their units.
3.  **Terrains**: Defines the terrain mapping.
4.  **Titles** (Optional): Assigns specific units to landed titles, overriding culture-based units.


## License
The original source code and all further edits of this repository fall under the GNU GENERAL PUBLIC LICENSE Version 3 license of distribution, and it was copied from the v1.2.2 version of the CW-Mappers distributed here inside the Crusader-Wars release (which is GPL3 licensed) https://github.com/farayC/Crusader-Wars/releases/tag/v1.0.14 and can be found in the `crusader-wars.zip` file within that Crusader-Wars v1.0.14 release.
