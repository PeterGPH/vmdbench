# gold_oracle_multi.tcl — portable "gold" metrics for ANY protein structure, headless.
#
#   GOLD_STRUCT=/path/to/structure.(pdb|cif) \
#     /Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64 -dispdev text -e gold_oracle_multi.tcl
#
# Emits parseable lines:  GOLD <key> <value>
# All metrics are structure-agnostic (no hard-coded residue ids), so adding a structure
# is just pointing GOLD_STRUCT at a new file. Each measurement is wrapped in catch{}.

proc emit {k v} { puts "GOLD $k $v" }

set structfile ""
if {[info exists env(GOLD_STRUCT)]} { set structfile $env(GOLD_STRUCT) }
if {$structfile eq "" || ![file exists $structfile]} {
    puts "GOLD_ERROR structfile-not-found '$structfile'"
    quit
}

set mol [mol new $structfile waitfor all]

# total atoms
if {[catch { emit natoms [molinfo $mol get numatoms] } err]} { puts "GOLD_ERROR natoms $err" }

# protein atom + residue counts
if {[catch {
    set p [atomselect $mol "protein"]
    emit nprotein [$p num]
    emit nresidues [llength [lsort -unique [$p get residue]]]
    $p delete
} err]} { puts "GOLD_ERROR protein_counts $err" }

# radius of gyration of the protein (Å)
if {[catch {
    set p [atomselect $mol "protein"]
    emit rgyr [format %.4f [measure rgyr $p]]
    $p delete
} err]} { puts "GOLD_ERROR rgyr $err" }

# RMSD of the structure vs itself = 0
if {[catch {
    set a [atomselect $mol "all"]
    emit rmsd_self [format %.4f [measure rmsd $a $a]]
    $a delete
} err]} { puts "GOLD_ERROR rmsd_self $err" }

# distance between the FIRST and LAST alpha-carbon (CA) atoms of the protein (Å)
if {[catch {
    set ca [atomselect $mol "protein and name CA"]
    set idx [$ca get index]
    $ca delete
    if {[llength $idx] >= 2} {
        set first [lindex $idx 0]
        set last  [lindex $idx end]
        emit ca_dist [format %.4f [measure bond [list $first $last]]]
    }
} err]} { puts "GOLD_ERROR ca_dist $err" }

# atom pairs within 8 Å among the protein (contacts; definition-sensitive)
if {[catch {
    set p [atomselect $mol "protein"]
    set res [measure contacts 8.0 $p]
    emit contacts8 [llength [lindex $res 0]]
    $p delete
} err]} { puts "GOLD_ERROR contacts8 $err" }

# solvent-accessible surface area of the protein, 1.4 Å probe (Å^2)
if {[catch {
    set p [atomselect $mol "protein"]
    emit sasa [format %.2f [measure sasa 1.4 $p]]
    $p delete
} err]} { puts "GOLD_ERROR sasa $err" }

quit
