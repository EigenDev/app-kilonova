use std::collections::HashMap;
use std::f64::consts::PI;
use ndarray::{ArcArray, Ix1, Ix2};
use serde::{Serialize, Deserialize};


/**
 * Type alias for a 2D block index
 */
pub type BlockIndex = (i32, usize);


/**
 * A struct to hold cached calculations of geometric primitives relevant to
 * spherical polar geometry
 */
#[derive(Clone)]
pub struct GridGeometry {
    pub radial_vertices:   ArcArray<f64, Ix1>,
    pub radial_face_areas: ArcArray<f64, Ix2>,
    pub polar_vertices:    ArcArray<f64, Ix1>,
    pub polar_face_areas:  ArcArray<f64, Ix2>,
    pub cell_volumes:      ArcArray<f64, Ix2>,
    pub cell_centers:      ArcArray<(f64, f64), Ix2>,
}


/**
 * A volume in (r, theta) space; polar section of a spherical annulus
 */
#[derive(Clone)]
pub struct SphericalPolarExtent {
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub lower_theta: f64,
    pub upper_theta: f64,
}


/**
 * A spherical polar extent, together with zone counts to make even subdivisions
 * in log-r and polar angle
 */
#[derive(Clone)]
pub struct SphericalPolarGrid {
    pub extent: SphericalPolarExtent,
    pub num_zones_r: usize,
    pub num_zones_q: usize,
}


/**
 * Abstract description of a spherical polar mesh
 */
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mesh {

    /// Inner boundary radius; the grid will start precisely here
    pub inner_radius: f64,

    /// Outer radius; the grid may extend up to one block past this
    pub outer_radius: f64,

    /// Number of zones from pole to pole
    pub num_polar_zones: usize,

    /// Number of radial zones in each block
    pub block_size: usize,
}




// ============================================================================
fn cell_volume(c0: (f64, f64), c1: (f64, f64)) -> f64
{
    let dcost = -(f64::cos(c1.1) - f64::cos(c0.1));
    2.0 * PI * (c1.0.powi(3) - c0.0.powi(3)) / 3.0 * dcost
}

fn face_area(c0: (f64, f64), c1: (f64, f64)) -> f64
{
    let s0 = c0.0 * f64::sin(c0.1);
    let s1 = c1.0 * f64::sin(c1.1);
    let z0 = c0.0 * f64::cos(c0.1);
    let z1 = c1.0 * f64::cos(c1.1);
    let ds = s1 - s0;
    let dz = z1 - z0;
    PI * (s0 + s1) * (ds * ds + dz * dz).sqrt()
}




// ============================================================================
impl SphericalPolarExtent {

    /**
     * Create a grid from this r-theta area with the given number of zones in the
     * polar and radial directions
     */
    pub fn grid(&self, num_zones_r: usize, num_zones_q: usize) -> SphericalPolarGrid {
        SphericalPolarGrid{
            extent: self.clone(),
            num_zones_r,
            num_zones_q,
        }
    }

    /**
     * Return the geometric centroid of this r-theta area
     */
    pub fn centroid(&self) -> (f64, f64) {
        let q = 0.5 * (self.lower_theta + self.upper_theta);
        let r = (self.inner_radius * self.outer_radius).sqrt();
        (r, q)
    }

    /**
     * Return the 3D volume of this extent
     */
    pub fn volume(&self) -> f64 {
        let c0 = (self.inner_radius, self.lower_theta);
        let c1 = (self.outer_radius, self.upper_theta);
        cell_volume(c0, c1)
    }

    /**
     * Return the area of the inner radial surface
     */
    pub fn face_area_r(&self) -> f64 {
        let c0 = (self.inner_radius, self.lower_theta);
        let c1 = (self.inner_radius, self.upper_theta);
        face_area(c0, c1)
    }

    /**
     * Return the area of the lower theta surface
     */
    pub fn face_area_q(&self) -> f64 {
        let c0 = (self.inner_radius, self.lower_theta);
        let c1 = (self.outer_radius, self.lower_theta);
        face_area(c0, c1)
    }
}




// ============================================================================
impl SphericalPolarGrid {

    /**
     * Return the r-theta vertex coordinate for index (i, j). For this function,
     * (i, j) are allowed to be outside the formal extent of this grid,
     * including being negative.
     */
    pub fn vertex_coordinate_signed(&self, i: i64, j: i64) -> (f64, f64) {
        let (y0, y1) = (self.extent.inner_radius.log(10.0), self.extent.outer_radius.log(10.0));
        let (q0, q1) = (self.extent.lower_theta, self.extent.upper_theta);
        let dy = (y1 - y0) / self.num_zones_r as f64;
        let dq = (q1 - q0) / self.num_zones_q as f64;
        let y = y0 + dy * i as f64;
        let q = q0 + dq * j as f64;
        (f64::powf(10.0, y), q)
    }

    /**
     * Return the r-theta coordinates of the vertex at the given index (i, j).
     */
    pub fn vertex_coordinate(&self, i: usize, j: usize) -> (f64, f64) {
        self.vertex_coordinate_signed(i as i64, j as i64)
    }

    /**
     * Return the spherical polar extent of a cell with index (i, j). For this
     * function, (i, j) are allowed to be outside the formal extent of this
     * grid, including being negative.
     */
    pub fn zone_signed(&self, i: i64, j: i64) -> SphericalPolarExtent {
        let lower = self.vertex_coordinate_signed(i + 0, j + 0);
        let upper = self.vertex_coordinate_signed(i + 1, j + 1);
        SphericalPolarExtent{
            inner_radius: lower.0,
            outer_radius: upper.0,
            lower_theta: lower.1,
            upper_theta: upper.1,
        }
    }

    /**
     * Return the spherical polar extent of a cell with the given index (i, j).
     */
    pub fn zone(&self, index: (usize, usize)) -> SphericalPolarExtent {
        self.zone_signed(index.0 as i64, index.1 as i64)
    }

    /**
     * Return a grid [`GridGeometry`] instance for this grid patch, containing
     * cached geometry data.
     */
    pub fn geometry(&self) -> GridGeometry {
        let nr = self.num_zones_r;
        let nq = self.num_zones_q;
        let radial_vertices   = ArcArray::from_shape_fn(nr + 1, |i| self.vertex_coordinate(i, 0).0);
        let polar_vertices    = ArcArray::from_shape_fn(nq + 1, |j| self.vertex_coordinate(0, j).1);
        let radial_face_areas = ArcArray::from_shape_fn((nr + 1, nq), |index| self.zone(index).face_area_r());
        let polar_face_areas  = ArcArray::from_shape_fn((nr, nq + 1), |index| self.zone(index).face_area_q());
        let cell_volumes      = ArcArray::from_shape_fn((nr, nq), |index| self.zone(index).volume());
        let cell_centers      = ArcArray::from_shape_fn((nr, nq), |index| self.zone(index).centroid());

        GridGeometry{
            radial_vertices,
            radial_face_areas,
            polar_vertices,
            polar_face_areas,
            cell_volumes,
            cell_centers,
        }
    }
}




// ============================================================================
impl Mesh {

    /**
     * Return the extent of the subgrid at this index
     */
    pub fn subgrid_extent(&self, index: BlockIndex) -> anyhow::Result<SphericalPolarExtent> {
        if self.outer_radius <= self.inner_radius {
            anyhow::bail!("outer_radius <= inner_radius")
        }
        let block_dlogr = self.block_size as f64 * std::f64::consts::PI / self.num_polar_zones as f64;
        Ok(SphericalPolarExtent{
            inner_radius: self.inner_radius * (1.0 + block_dlogr).powf(index.0 as f64),
            outer_radius: self.inner_radius * (1.0 + block_dlogr).powf(index.0 as f64 + 1.0),
            lower_theta: 0.0,
            upper_theta: std::f64::consts::PI,
        })
    }

    /**
     * Return the subgrid object at the given index
     */
    pub fn subgrid(&self, index: BlockIndex) -> anyhow::Result<SphericalPolarGrid> {
        Ok(self.subgrid_extent(index)?.grid(self.block_size, self.num_polar_zones))
    }

    /**
     * Return a map of the subgrid objects on this mesh
     */
    pub fn grid_blocks(&self) -> anyhow::Result<HashMap<BlockIndex, SphericalPolarGrid>> {
        let mut blocks = HashMap::new();
        for i in 0.. {
            let index = (i, 0);
            let extent = self.subgrid_extent(index)?;

            if extent.inner_radius >= self.outer_radius {
                break
            } else {
                blocks.insert(index, extent.grid(self.block_size, self.num_polar_zones));
            }
        }
        Ok(blocks)
    }

    /**
     * Return a map of the subgrid geometry objects on this mesh (for convenience)
     */
    pub fn grid_blocks_geometry(&self) -> anyhow::Result<HashMap<BlockIndex, GridGeometry>> {
        Ok(self.grid_blocks()?
            .iter()
            .map(|(&index, grid)| (index, grid.geometry()))
            .collect())
    }
}
