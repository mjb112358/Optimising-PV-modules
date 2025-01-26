import numpy as np
import math
import matplotlib.pyplot as plt
import csv


# Part 1 : The first part verifies the mathematics behind the reflection 
# calculations. A graph is generated to validate the calculations.


# Part 2 : This section focuses on finding the optimal radii and PV cell 
# positions. The function used in the previous section is retained, but 
# with some parameters now called externally rather than embedded within 
# the function. The goal is to create a loop over various radii, positions, 
# incident ray angles, and cell displacement errors, the latter being a new 
# parameter that accounts for manufacturing inaccuracies in PV cell placement.


def plot_convex_mirror_reflection_part1(
    y_max_reflector, mirror_radius, ray_angle_deg, 
    num_rays, object_width, object_x, object_y
):
    """
    Plots the convex mirror, incoming rays, and reflected rays.
    Returns the percentage of rays (out of those that actually
    reach the mirror) that hit the object after reflection.
    """
    # Convert angle to radians
    ray_angle_rad = math.radians(ray_angle_deg)
    
    # Mirror specifications
    max_theta = math.asin(y_max_reflector / mirror_radius)
    
    # Generate points on the mirror's circumference
    theta_vals = np.linspace(-max_theta, max_theta, 400)
    x_mirror = mirror_radius * np.cos(theta_vals)
    y_mirror = mirror_radius * np.sin(theta_vals)
    
    # Set up the figure
    plt.figure()
    plt.title(f"Reflection on a Convex Mirror (Angle = {ray_angle_deg}°)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axis("equal")
    plt.grid(True)
    
    # Plot the mirror
    # plt.plot(x_mirror, y_mirror, "k", label="Convex Mirror")
    
    # Determine the mirror-intersection points of the incoming rays
    ray_positions = np.linspace(-max_theta, max_theta, num_rays)
    x_ray_intersects = mirror_radius * np.cos(ray_positions)
    y_ray_intersects = mirror_radius * np.sin(ray_positions)
    
    ray_start_x = -1000  # Start far left to simulate "incoming from infinity"
    
    number_of_hits = 0
    rays_reaching_mirror = 0
    
    for idx in range(num_rays):
        # Where the ray meets the mirror
        ray_end_x = x_ray_intersects[idx]
        ray_end_y = y_ray_intersects[idx]
        
        # Calculate the ray's start y given the angle
        slope_incoming = math.tan(ray_angle_rad)
        ray_start_y = ray_end_y - slope_incoming * (ray_end_x - ray_start_x)
        
        # Check if incoming ray directly hits the object (bypassing the mirror)
        y_at_object_x_incoming = ray_start_y + slope_incoming * (object_x - ray_start_x)
        
        if min(object_y) <= y_at_object_x_incoming <= max(object_y):
            # It hits the object directly, skip reflection counting
            continue
        
        # This ray actually reaches the mirror
        rays_reaching_mirror += 1
        
        # Plot the incoming ray
        #plt.plot(
        #    [ray_start_x, ray_end_x],
        #    [ray_start_y, ray_end_y],
        #    "k--"
        #)
        
        # Compute reflection:
        normal_angle = math.atan2(ray_end_y, ray_end_x)
        incident_angle = ray_angle_rad - normal_angle
        reflected_angle = normal_angle - incident_angle  # Law of reflection
        
        # Extend the reflected ray
        reflected_x_far = ray_end_x - 1000 * math.cos(reflected_angle)
        reflected_y_far = ray_end_y - 1000 * math.sin(reflected_angle)
        
        #plt.plot(
        #    [ray_end_x, reflected_x_far],
        #    [ray_end_y, reflected_y_far],
        #    color=(1, 0.5, 0),  # orange
        #    label="Reflected Ray" if idx == 0 else ""
        #)
        
        # Check if the reflected ray hits the object
        distance_to_object = object_x - ray_end_x
        slope_reflected = math.tan(reflected_angle)
        y_at_object_x_reflected = ray_end_y + slope_reflected * distance_to_object
        
        if min(object_y) <= y_at_object_x_reflected <= max(object_y):
            number_of_hits += 1
            # Plot the hit point on the object in green
            #plt.plot(object_x, y_at_object_x_reflected, "go")
    
    # Plot the object (vertical line at object_x)
    #plt.plot(
        #[object_x, object_x],
        #[min(object_y), max(object_y)],
        #"b-",
        #linewidth=2,
        #label="Object"
    #)
    
    #plt.legend()
    #plt.show()
    
    # Calculate percentage of hits among rays that reached the mirror
    if rays_reaching_mirror > 0:
        percentage_hits = (number_of_hits / rays_reaching_mirror) * 100
    else:
        percentage_hits = 0.0
    
    return percentage_hits


def part_one():
    """
    Replicates the first portion of the MATLAB script:
      - Loops over angles 0° to 10°
      - Plots rays & mirror
      - Prints the percentage of rays hitting the object.
    """
    # Parameters
    y_max_reflector = 59.5
    mirror_radius = 117
    num_rays = 301
    
    # Object specs
    object_width = 17
    object_x = 66
    object_y = [-object_width/2, object_width/2]
    
    ray_angles_deg = range(0, 11)  # 0° to 10°
    
    for angle_deg in ray_angles_deg:
        percentage_hits = plot_convex_mirror_reflection_part1(
            y_max_reflector,
            mirror_radius,
            angle_deg,
            num_rays,
            object_width,
            object_x,
            object_y
        )
        print(f"Angle {angle_deg}° - Percentage of rays that hit the object: {percentage_hits:.2f}%")

# ----------------------------------------------------------------------
# Part 2: Sweeping positions/radii and writing results to CSV
# ----------------------------------------------------------------------

def compute_convex_mirror_reflection_part2(
    y_max_reflector, mirror_radius, ray_angle_deg,
    num_rays, object_x, object_width
):
    """
    Computes the percentage (hits) of rays that strike the object
    after reflecting off a convex mirror. Returns:
        hits_percentage (float): % of rays that hit the object (among those reaching mirror)
        ray_count (int): number of rays that actually reached the mirror
    """
    ray_angle_rad = math.radians(ray_angle_deg)
    
    # Mirror specs
    max_theta = math.asin(y_max_reflector / mirror_radius)
    
    # Object y-range
    object_y = [-object_width/2, object_width/2]
    
    # Discretize the mirror intersection points
    ray_positions = np.linspace(-max_theta, max_theta, num_rays)
    x_ray_intersects = mirror_radius * np.cos(ray_positions)
    y_ray_intersects = mirror_radius * np.sin(ray_positions)
    
    direct_hits = 0  # Rays that directly hit the object (no reflection)
    reflected_hits = 0  # Rays that hit the object after reflection
    
    for idx in range(num_rays):
        rx = x_ray_intersects[idx]
        ry = y_ray_intersects[idx]
        
        slope_incoming = math.tan(ray_angle_rad)
        # Start from far left: x = -1000
        # y_start = ry - slope_incoming*(rx + 1000)
        y_start = ry - slope_incoming * (rx + 1000)
        
        # Check direct hit
        # y_at_object_x = y_start + slope_incoming*(object_x + 1000)
        y_object_incoming = y_start + slope_incoming * (object_x + 1000)
        if min(object_y) <= y_object_incoming <= max(object_y):
            direct_hits += 1
            continue
        
        # Reflection
        normal_angle = math.atan2(ry, rx)
        incident_angle = ray_angle_rad - normal_angle
        reflected_angle = normal_angle - incident_angle
        
        # Intersection with x=object_x after reflection
        distance_to_object = object_x - rx
        slope_reflected = math.tan(reflected_angle)
        y_object_reflected = ry + slope_reflected * distance_to_object
        
        if min(object_y) <= y_object_reflected <= max(object_y):
            reflected_hits += 1
    
    ray_count = num_rays - direct_hits
    if ray_count > 0:
        hits_percentage = (reflected_hits / ray_count) * 100.0
    else:
        hits_percentage = 0.0
    
    return hits_percentage, ray_count


def part_two():
    """
    Sweeps over angles, object positions, and mirror radii, and
    writes optimal configurations to a CSV file.
    """
    # Parameters
    y_max_reflector = 59
    num_rays = 101
    object_width = 17
    
    # Angles in degrees
    ray_angle_deg_list = range(0, 11)  # 0° to 10°
    
    # Ranges for the object position and mirror radius
    mirror_radii = range(100, 351)      # 100 to 350
    object_positions = range(50, 201)   # 50 to 200
    
    # Dictionary: angle -> list of (position, radius, hits_percentage)
    optimal_configurations = {}
    
    for angle_deg in ray_angle_deg_list:
        max_hits = 0.0
        configs_for_angle = []
        
        for pos in object_positions:
            for rad in mirror_radii:
                hits_percentage, _ = compute_convex_mirror_reflection_part2(
                    y_max_reflector, rad, angle_deg, num_rays, pos, object_width
                )
                
                if hits_percentage > max_hits:
                    max_hits = hits_percentage
                    configs_for_angle = [(pos, rad, hits_percentage)]
                elif abs(hits_percentage - max_hits) < 1e-9:
                    # If it's exactly equal to the current max, add to the list
                    configs_for_angle.append((pos, rad, hits_percentage))
        
        optimal_configurations[angle_deg] = configs_for_angle
    
    # --- Write results to CSV instead of printing ---
    csv_filename = ".../optimal_configurations_part2.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["Angle (deg)", "Position (mm)", "Radius (mm)", "Hits (%)"])
        
        # Rows: one per optimal configuration
        for angle_deg, configs in optimal_configurations.items():
            for (pos, rad, hits) in configs:
                writer.writerow([angle_deg, pos, rad, hits])
    
    print(f"Optimal configurations saved to: {csv_filename}")


def main():
    print("=== Part 1: Plotting Rays & Mirror ===")
    part_one()
    
    print("\n=== Part 2: Sweeping Positions & Radii, Writing CSV ===")
    part_two()


if __name__ == "__main__":
    main()
