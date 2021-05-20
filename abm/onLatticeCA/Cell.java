// ========================================================================
// Definition of a single cell.
// ========================================================================

package onLatticeCA;

import Framework.GridsAndAgents.AgentList;
import Framework.GridsAndAgents.AgentSQ2Dunstackable;
import Framework.Util;

// ========================================================================
// Class to model individual cells
class Cell extends AgentSQ2Dunstackable<OnLatticeCA > {
    double divisionRate;
    double movementRate;
    double deathRate;
    int resistance; // 0 = sensitive, 1 = resistant

    // -----------------------------------
    boolean HasSpace(){
        boolean successfulDivisionB = false;
        return MapEmptyHood(G.hood)>0;
    }
    // -----------------------------------
    void Divide(){
        int nOpts = MapEmptyHood(G.hood); // Finds von neumann neighborhood indices around cell.
        int iDaughter = G.hood[G.rn.Int(nOpts)]; // Choose an option at random
        Cell daughter = G.NewAgentSQ(iDaughter); // Generate a daughter, the other is technically the original cell
        // Inherit division, movement and resistance characteristics (assuming no mutation here).
        daughter.divisionRate = divisionRate;
        daughter.movementRate = movementRate;
        daughter.deathRate = deathRate;
        daughter.resistance=resistance;
        daughter.Draw();
    }

    // -----------------------------------
    boolean Move(){
        boolean successfulMoveB = false;
        int nOpts=MapEmptyHood(G.hood);// identify the empty spots in the cell's neighbourhood (if there are any).
        if(nOpts>0){
            int iDestination = G.hood[G.rn.Int(nOpts)];
            MoveSafeSQ(G.ItoX(iDestination), G.ItoY(iDestination));
            successfulMoveB = true;
        }
        return successfulMoveB;
    }

    // -----------------------------------
    // Draws sensitive and resistant cells in different colours.
    // For colours from graphs: Util.RGB256(17, 91, 28): Util.RGB256(132, 18, 10)
    // For red/blue combination: Util.RGB256(26, 133, 255): Util.RGB256(212, 17, 89))
    void Draw(){
        G.vis.SetPix(Isq(), (resistance==0)? Util.RGB256(117, 197, 114): Util.RGB256(216, 27, 96));//sets a single pixel
//        G.vis.SetPix(Isq(), (resistance==0)? Util.CategorialColor(13): Util.CategorialColor(11));//sets a single pixel
    }

}