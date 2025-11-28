# CC-Mappers
Mappers for [Crusader Conflicts](https://github.com/szmania/Crusader-Wars/releases/latest), Community Successor for Crusader Wars mod.

## Mapper Loading Order

The Crusader Conflicts mod loads culture, faction, and terrain mapping XML files in a specific order to allow for overrides. The loading process is as follows:

1.  **Base Mappers**: The official playthrough base mappers are loaded first. These files are identified by the naming convention `OfficialCC_*.xml`.
2.  **Optional Add-on Mappers**: After the base mappers, optional add-on files that are not submod files are loaded in alphabetical order.
3.  **Submod Mappers**: After the optional add-ons, submod mappers are loaded. These are identified by a `submod_tag` attribute in the root element of the XML file.
4.  **Submod Optional Add-on Mappers**: Finally, optional add-ons for submods are loaded in alphabetical order. These files are identified by a `submod_addon_tag` attribute in the root element, which links them to a specific submod.

This loading order means that files loaded later will overwrite mappings from files loaded earlier. For example, a submod mapper can override a base mapper, and a submod's optional add-on can override the submod's own mappings. When multiple files of the same type are loaded (like optional add-ons), the one that comes last alphabetically takes precedence.
The files within the `Cultures`, `Factions`, `Terrains`, and `Titles` directories are loaded in the same way, and optional add ons and submod optional addons are loaded alphabetically within their respective loading group. If you have multiple files in these directories, their contents will be merged.


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


## License
The original source code and all further edits of this repository fall under the GNU GENERAL PUBLIC LICENSE Version 3 license of distribution, and it was copied from the v1.2.2 version of the CW-Mappers distributed here inside the Crusader-Wars release (which is GPL3 licensed) https://github.com/farayC/Crusader-Wars/releases/tag/v1.0.14 and can be found in the `crusader-wars.zip` file within that Crusader-Wars v1.0.14 release.
