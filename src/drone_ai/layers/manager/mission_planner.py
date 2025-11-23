"""
Mission Planner Module for Drone AI.

Handles high-level decision making: what to deliver, where, when, and how to
optimize multi-delivery operations. Sits ABOVE path selection which handles
waypoint-to-waypoint navigation.

Hierarchy: Mission Planner -> Path Selection -> Flight Control
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from drone_ai.core.config import DroneSpecs


@dataclass
class DeliveryRequest:
    """A single delivery task."""
    id: int
    pickup_position: np.ndarray      # Where to pick up package
    dropzone_position: np.ndarray    # Where to deliver
    priority: int = 1                # 1=normal, 2=urgent, 3=critical
    time_window: Optional[Tuple[float, float]] = None  # (earliest, latest) in seconds
    weight: float = 1.0              # Package weight in kg
    reward_value: float = 100.0      # Points for successful delivery

    def __post_init__(self):
        """Ensure positions are numpy arrays."""
        if not isinstance(self.pickup_position, np.ndarray):
            self.pickup_position = np.array(self.pickup_position, dtype=np.float32)
        if not isinstance(self.dropzone_position, np.ndarray):
            self.dropzone_position = np.array(self.dropzone_position, dtype=np.float32)


@dataclass
class MissionState:
    """Current mission status."""
    pending_deliveries: List[DeliveryRequest] = field(default_factory=list)
    completed_deliveries: List[int] = field(default_factory=list)  # IDs
    failed_deliveries: List[int] = field(default_factory=list)     # IDs
    current_delivery: Optional[DeliveryRequest] = None
    battery_level: float = 1.0              # 0.0 to 1.0
    total_score: float = 0.0
    elapsed_time: float = 0.0

    def reset(self):
        """Reset mission state for new episode."""
        self.pending_deliveries = []
        self.completed_deliveries = []
        self.failed_deliveries = []
        self.current_delivery = None
        self.battery_level = 1.0
        self.total_score = 0.0
        self.elapsed_time = 0.0


class MissionPlanner:
    """
    High-level mission planner for multi-delivery drone operations.

    Responsibilities:
    - Managing delivery queue
    - Selecting optimal next delivery
    - Battery management decisions
    - Route optimization (TSP-like)
    - Priority-based scheduling

    Uses physics-based calculations for:
    - Flight time estimation
    - Battery consumption
    - Safe travel distance with payload
    """

    # Battery thresholds (fraction of usable battery)
    BATTERY_CRITICAL = 0.15    # Must return to base
    BATTERY_LOW = 0.30         # Consider returning
    BATTERY_SAFE = 0.50        # Safe for new deliveries

    # Cost factors
    DISTANCE_COST_FACTOR = 0.001  # Battery cost per meter
    DELIVERY_OVERHEAD = 0.05       # Fixed cost per delivery operation

    def __init__(
        self,
        base_position: np.ndarray,
        battery_capacity: float = 1000.0,
        drone_specs: Optional[DroneSpecs] = None
    ):
        """
        Initialize the mission planner.

        Args:
            base_position: Home base position for pickups and recharging
            battery_capacity: Maximum battery capacity in abstract units (legacy)
            drone_specs: Physical drone specifications for realistic calculations
        """
        if not isinstance(base_position, np.ndarray):
            base_position = np.array(base_position, dtype=np.float32)
        self.base_position = base_position
        self.battery_capacity = battery_capacity
        self.drone_specs = drone_specs or DroneSpecs()
        self.state = MissionState()
        self._delivery_id_counter = 0

    def reset(self):
        """Reset planner state for new episode."""
        self.state.reset()
        self._delivery_id_counter = 0

    def add_delivery(self, request: DeliveryRequest) -> None:
        """Add a new delivery request to the queue."""
        self.state.pending_deliveries.append(request)
        self._sort_pending_deliveries()

    def create_delivery(
        self,
        dropzone_position: np.ndarray,
        priority: int = 1,
        time_window: Optional[Tuple[float, float]] = None,
        weight: float = 1.0,
        reward_value: float = 100.0
    ) -> DeliveryRequest:
        """
        Create and add a new delivery request.

        Args:
            dropzone_position: Delivery destination
            priority: 1=normal, 2=urgent, 3=critical
            time_window: Optional (earliest, latest) time bounds
            weight: Package weight in kg
            reward_value: Points for successful delivery

        Returns:
            The created DeliveryRequest
        """
        request = DeliveryRequest(
            id=self._delivery_id_counter,
            pickup_position=self.base_position.copy(),
            dropzone_position=np.array(dropzone_position, dtype=np.float32),
            priority=priority,
            time_window=time_window,
            weight=weight,
            reward_value=reward_value
        )
        self._delivery_id_counter += 1
        self.add_delivery(request)
        return request

    def _sort_pending_deliveries(self) -> None:
        """Sort pending deliveries by priority (descending) and urgency."""
        def sort_key(d: DeliveryRequest) -> Tuple[int, float, float]:
            deadline = d.time_window[1] if d.time_window else float('inf')
            current_pos = (
                self.state.current_delivery.dropzone_position
                if self.state.current_delivery
                else self.base_position
            )
            distance = np.linalg.norm(d.dropzone_position - current_pos)
            return (-d.priority, deadline, distance)

        self.state.pending_deliveries.sort(key=sort_key)

    def select_next_delivery(self) -> Optional[DeliveryRequest]:
        """
        Choose optimal next delivery based on priority, distance, and battery.

        Returns:
            Next delivery to attempt, or None if no viable delivery
        """
        if not self.state.pending_deliveries:
            return None

        self._sort_pending_deliveries()

        for delivery in self.state.pending_deliveries:
            cost = self.estimate_delivery_cost(delivery)
            return_cost = self._estimate_return_cost(delivery.dropzone_position)

            if self.state.battery_level >= cost + return_cost + self.BATTERY_CRITICAL:
                self.state.pending_deliveries.remove(delivery)
                self.state.current_delivery = delivery
                return delivery

        return None

    def complete_current_delivery(self, success: bool) -> float:
        """
        Mark current delivery as complete.

        Args:
            success: Whether delivery was successful

        Returns:
            Score gained/lost from this delivery
        """
        if not self.state.current_delivery:
            return 0.0

        delivery = self.state.current_delivery
        score = 0.0

        if success:
            self.state.completed_deliveries.append(delivery.id)
            score = delivery.reward_value
            score += delivery.priority * 25.0

            if delivery.time_window:
                if self.state.elapsed_time <= delivery.time_window[1]:
                    score += 30.0
                else:
                    score -= 20.0
        else:
            self.state.failed_deliveries.append(delivery.id)
            score = -delivery.reward_value * 0.5

        self.state.total_score += score
        self.state.current_delivery = None
        return score

    def should_return_to_base(self) -> bool:
        """Decide if drone should return for battery/reload."""
        if self.state.battery_level <= self.BATTERY_CRITICAL:
            return True

        if not self.state.pending_deliveries:
            return True

        if self.state.battery_level <= self.BATTERY_LOW:
            for delivery in self.state.pending_deliveries:
                cost = self.estimate_delivery_cost(delivery)
                return_cost = self._estimate_return_cost(delivery.dropzone_position)
                if self.state.battery_level >= cost + return_cost + self.BATTERY_CRITICAL:
                    return False
            return True

        return False

    def estimate_delivery_cost(self, delivery: DeliveryRequest) -> float:
        """Estimate battery/time cost for a delivery."""
        current_pos = (
            self.state.current_delivery.dropzone_position
            if self.state.current_delivery
            else self.base_position
        )

        to_pickup = np.linalg.norm(delivery.pickup_position - current_pos)
        to_dropzone = np.linalg.norm(
            delivery.dropzone_position - delivery.pickup_position
        )

        total_distance = to_pickup + to_dropzone
        cost = total_distance * self.DISTANCE_COST_FACTOR

        weight_multiplier = 1.0 + (delivery.weight - 1.0) * 0.1
        cost *= weight_multiplier
        cost += self.DELIVERY_OVERHEAD

        return min(cost, 1.0)

    def _estimate_return_cost(self, from_position: np.ndarray) -> float:
        """Estimate cost to return to base from a position."""
        distance = np.linalg.norm(self.base_position - from_position)
        return distance * self.DISTANCE_COST_FACTOR + self.DELIVERY_OVERHEAD * 0.5

    def optimize_delivery_order(self) -> List[DeliveryRequest]:
        """Reorder pending deliveries for optimal route (TSP-like)."""
        if len(self.state.pending_deliveries) <= 1:
            return self.state.pending_deliveries.copy()

        remaining = self.state.pending_deliveries.copy()
        optimized = []

        current_pos = (
            self.state.current_delivery.dropzone_position
            if self.state.current_delivery
            else self.base_position
        )

        while remaining:
            best_delivery = None
            best_score = float('inf')

            for delivery in remaining:
                distance = np.linalg.norm(delivery.dropzone_position - current_pos)
                priority_factor = 1.0 / (delivery.priority + 0.5)

                urgency_factor = 1.0
                if delivery.time_window:
                    time_remaining = delivery.time_window[1] - self.state.elapsed_time
                    if time_remaining > 0:
                        urgency_factor = min(1.0, time_remaining / 60.0)

                score = distance * priority_factor * urgency_factor

                if score < best_score:
                    best_score = score
                    best_delivery = delivery

            if best_delivery:
                remaining.remove(best_delivery)
                optimized.append(best_delivery)
                current_pos = best_delivery.dropzone_position

        return optimized

    def update_battery(self, new_level: float) -> None:
        """Update the tracked battery level."""
        self.state.battery_level = max(0.0, min(1.0, new_level))

    def update_time(self, delta_time: float) -> None:
        """Update elapsed mission time."""
        self.state.elapsed_time += delta_time

    def get_mission_summary(self) -> dict:
        """Get a summary of current mission status."""
        return {
            'pending_count': len(self.state.pending_deliveries),
            'completed_count': len(self.state.completed_deliveries),
            'failed_count': len(self.state.failed_deliveries),
            'current_delivery_id': (
                self.state.current_delivery.id
                if self.state.current_delivery
                else None
            ),
            'battery_level': self.state.battery_level,
            'total_score': self.state.total_score,
            'elapsed_time': self.state.elapsed_time,
            'mission_complete': (
                len(self.state.pending_deliveries) == 0
                and self.state.current_delivery is None
            )
        }

    def get_nearest_delivery_distance(self) -> float:
        """Get distance to the nearest pending delivery."""
        if not self.state.pending_deliveries:
            return 0.0

        current_pos = (
            self.state.current_delivery.dropzone_position
            if self.state.current_delivery
            else self.base_position
        )

        min_distance = float('inf')
        for delivery in self.state.pending_deliveries:
            distance = np.linalg.norm(delivery.dropzone_position - current_pos)
            min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else 0.0

    def get_highest_priority(self) -> int:
        """Get the highest priority among pending deliveries."""
        if not self.state.pending_deliveries:
            return 0
        return max(d.priority for d in self.state.pending_deliveries)

    # =========================================================================
    # Physics-Based Calculations
    # =========================================================================

    def get_max_range(self, payload_kg: float = 0.0) -> float:
        """Calculate maximum one-way flight range with given payload."""
        specs = self.drone_specs
        usable_battery = specs.battery_capacity_wh * (1 - specs.reserve_battery)
        cruise_power = specs.cruise_power(payload_kg)
        flight_time_hours = usable_battery / cruise_power
        max_range = specs.cruise_speed * flight_time_hours * 3600
        return max_range

    def get_safe_range(self, payload_kg: float = 0.0) -> float:
        """Calculate safe round-trip range with given payload."""
        specs = self.drone_specs
        usable_battery = specs.battery_capacity_wh * (1 - specs.reserve_battery)
        hover_energy = specs.hover_power(payload_kg) * (60 / 3600)
        usable_battery -= hover_energy

        outbound_power = specs.cruise_power(payload_kg)
        return_power = specs.cruise_power(0)

        safe_range = (usable_battery * specs.cruise_speed * 3600) / (outbound_power + return_power)
        return max(0, safe_range)

    def get_flight_time(
        self,
        distance: float,
        payload_kg: float = 0.0,
        include_vertical: bool = True,
        altitude_change: float = 0.0
    ) -> float:
        """Estimate flight time for a given distance and payload."""
        specs = self.drone_specs
        horizontal_time = distance / specs.cruise_speed

        vertical_time = 0.0
        if include_vertical and altitude_change != 0:
            if altitude_change > 0:
                vertical_time = altitude_change / specs.climb_rate
            else:
                vertical_time = abs(altitude_change) / specs.descent_rate

        return horizontal_time + vertical_time

    def get_energy_cost(
        self,
        distance: float,
        payload_kg: float = 0.0,
        altitude_change: float = 0.0
    ) -> float:
        """Calculate energy cost for a flight segment in Watt-hours."""
        specs = self.drone_specs

        horizontal_time_hours = (distance / specs.cruise_speed) / 3600
        cruise_energy = specs.cruise_power(payload_kg) * horizontal_time_hours

        vertical_energy = 0.0
        if altitude_change > 0:
            climb_time_hours = (altitude_change / specs.climb_rate) / 3600
            vertical_energy = specs.climb_power(payload_kg) * climb_time_hours
        elif altitude_change < 0:
            descent_time_hours = (abs(altitude_change) / specs.descent_rate) / 3600
            vertical_energy = specs.hover_power(payload_kg) * descent_time_hours

        return cruise_energy + vertical_energy

    def can_complete_delivery(
        self,
        delivery: DeliveryRequest,
        current_position: Optional[np.ndarray] = None,
        current_battery: Optional[float] = None
    ) -> Tuple[bool, dict]:
        """Check if a delivery can be safely completed."""
        if current_position is None:
            current_position = (
                self.state.current_delivery.dropzone_position
                if self.state.current_delivery
                else self.base_position
            )
        if current_battery is None:
            current_battery = self.state.battery_level

        to_pickup = np.linalg.norm(delivery.pickup_position - current_position)
        to_dropzone = np.linalg.norm(
            delivery.dropzone_position - delivery.pickup_position
        )
        to_base = np.linalg.norm(self.base_position - delivery.dropzone_position)

        alt_to_pickup = delivery.pickup_position[2] - current_position[2]
        alt_to_dropzone = delivery.dropzone_position[2] - delivery.pickup_position[2]
        alt_to_base = self.base_position[2] - delivery.dropzone_position[2]

        energy_to_pickup = self.get_energy_cost(to_pickup, 0, alt_to_pickup)
        energy_to_dropzone = self.get_energy_cost(
            to_dropzone, delivery.weight, alt_to_dropzone
        )
        energy_to_base = self.get_energy_cost(to_base, 0, alt_to_base)

        hover_overhead = (
            self.drone_specs.hover_power(delivery.weight) * (30 / 3600) +
            self.drone_specs.hover_power(0) * (30 / 3600)
        )

        total_energy = energy_to_pickup + energy_to_dropzone + energy_to_base + hover_overhead

        available_wh = current_battery * self.drone_specs.battery_capacity_wh
        reserve_wh = self.drone_specs.reserve_battery * self.drone_specs.battery_capacity_wh
        usable_wh = available_wh - reserve_wh

        time_to_pickup = self.get_flight_time(to_pickup, 0, True, alt_to_pickup)
        time_to_dropzone = self.get_flight_time(
            to_dropzone, delivery.weight, True, alt_to_dropzone
        )
        time_to_base = self.get_flight_time(to_base, 0, True, alt_to_base)
        total_time = time_to_pickup + time_to_dropzone + time_to_base + 60

        details = {
            'can_complete': usable_wh >= total_energy,
            'energy_required_wh': total_energy,
            'energy_available_wh': usable_wh,
            'battery_cost_percent': total_energy / self.drone_specs.battery_capacity_wh,
            'margin_wh': usable_wh - total_energy,
            'estimated_time_seconds': total_time,
            'distance_total_m': to_pickup + to_dropzone + to_base,
        }

        return details['can_complete'], details
